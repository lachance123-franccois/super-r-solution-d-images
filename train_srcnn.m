%TRAIN_SRCNN Entraîne le réseau SRCNN x3 from scratch, sans toolbox.
%   Ce script réalise l'entraînement d'un réseau convolutif à 3 couches.
%
%   Méthodologie :
%   - Patchs 33x33 tirés au hasard dans les paires (MR, HR) des images d'entraînement.
%   - Augmentation de données : 8 symétries (rotations de 90° et inversions miroir).
%   - Taille de mini-batch : 16 exemples.
%   - Optimisateur : Adam codé à la main (pas initial de 1e-3, réduit à 1e-4 à 70% de l'apprentissage).
%   - Perte : Erreur Quadratique Moyenne (MSE) calculée uniquement sur le centre 21x21
%             du patch pour exclure les incertitudes dues au rembourrage de zéros aux bords.
%   - Variante résiduelle : Le réseau prédit le résidu HR - MR, offrant une convergence
%                           beaucoup plus rapide que la méthode directe classique.

clear; rng(0); % Fixation de la graine aléatoire pour la reproductibilité

% --- Configuration et hyperparamètres ---
scale  = 3;     % Facteur de super-résolution
P      = 33;    % Taille des patchs carrés d'entraînement (33x33)
B      = 16;    % Taille du mini-batch (nombre de patchs par itération)
steps  = 12000; % Nombre total d'itérations d'entraînement
lr0    = 1e-3;  % Taux d'apprentissage initial (learning rate)
trainDir = fullfile('data', 'train');

% --- Étape 1 : Préparation des paires d'images (MR, HR) ---
files = dir(fullfile(trainDir, '*.png'));
nFiles = numel(files);
pairs = cell(nFiles, 2);

fprintf('--- Chargement et dégradation de la base d''entraînement (%d images) ---\n', nFiles);
for f = 1:nFiles
    % Chargement et conversion de l'image haute résolution d'origine
    hr = to_gray(read_image(fullfile(trainDir, files(f).name)));
    
    % Dégradation pour obtenir l'image basse résolution
    [lr, hr] = degrade(hr, scale);
    
    % Interpolation géométrique initiale (Moyenne Résolution) et stockage des paires
    pairs{f, 1} = resize_bicubic(lr, size(hr));
    pairs{f, 2} = hr;
end

% --- Étape 2 : Initialisation des paramètres du réseau (Méthode de He / Xavier) ---
net.f = [9 1 5];       % Tailles des fenêtres de filtres pour les 3 couches convolutives
dims  = [1 64 32 1];   % Nombre de canaux d'entrée/sortie pour chaque couche [Cin, C1, C2, Cout]

for l = 1:3
    fan_in = net.f(l)^2 * dims(l); % Nombre de connexions d'entrée par neurone
    if l < 3
        % Initialisation de He (msra) adaptée aux couches suivies d'une activation ReLU
        net.W{l} = sqrt(2 / fan_in) * randn(fan_in, dims(l + 1));
    else
        % Initialisation fine pour la couche finale (départ proche de l'identité / résidu nul)
        net.W{l} = 1e-3 * randn(fan_in, dims(l + 1));    
    end
    net.b{l} = zeros(1, dims(l + 1)); % Initialisation des biais à zéro
end

% --- Étape 3 : Initialisation des moments pour l'optimisateur Adam ---
mW = cellfun(@(w) zeros(size(w)), net.W, 'UniformOutput', false); % Premier moment (moyenne du gradient)
vW = mW;                                                         % Second moment (variance non centrée)
mb = cellfun(@(b) zeros(size(b)), net.b, 'UniformOutput', false); 
vb = mb;

% Chargement de l'image de test 'cameraman' pour assurer le suivi visuel des performances
cam = to_gray(read_image(fullfile('data', 'test', 'camera.png')));
[camLR, camHR] = degrade(cam, scale);

% Définition de la fenêtre centrale 21x21 pour exclure les artefacts de bord (33 - 2*6 = 21)
c = 7 : P-6;                                   
history = zeros(steps, 1); % Historique de la fonction de perte
tic;

fprintf('--- Début de l''entraînement du réseau SRCNN (%d itérations) ---\n', steps);
% --- Boucle principale d'optimisation ---
for it = 1:steps
    % Planification du taux d'apprentissage (Learning Rate Decay à 70% de l'entraînement)
    lr = lr0;
    if it >= 0.7 * steps
        lr = lr0 / 10; 
    end
    
    % Allocation du tenseur 4D de mini-batch pour les entrées (X) et les cibles (Y)
    X = zeros(P, P, 1, B); 
    Y = X;
    
    % --- Étape 4 : Échantillonnage aléatoire et Augmentation de données ---
    for n = 1:B
        % Sélection d'une image d'entraînement au hasard
        f = randi(size(pairs, 1));
        mr = pairs{f, 1}; 
        hr = pairs{f, 2};
        
        % Tirage aléatoire des coordonnées d'un patch de taille P x P
        i = randi(size(hr, 1) - P + 1); 
        j = randi(size(hr, 2) - P + 1);
        a = mr(i:i+P-1, j:j+P-1); 
        b = hr(i:i+P-1, j:j+P-1);
        
        % Augmentation géométrique : Rotations aléatoires de 90°
        rot = randi(4) - 1;
        a = rot90(a, rot); 
        b = rot90(b, rot);
        
        % Augmentation géométrique : Inversion miroir horizontale aléatoire (Flipping)
        if rand < 0.5
            a = fliplr(a); 
            b = fliplr(b); 
        end
        
        % Assignation dans le tenseur du mini-batch
        X(:, :, 1, n) = a; 
        Y(:, :, 1, n) = b;
    end
    
    % --- Étape 5 : Passe avant (Forward propagation) ---
    [out, cache] = srcnn_forward(net, X);
    
    % --- Étape 6 : Évaluation de l'erreur (MSE sur le centre du patch) ---
    d = zeros(size(out));
    d(c, c, :, :) = out(c, c, :, :) - Y(c, c, :, :); % Calcul de l'écart sur la zone utile
    dc = d(c, c, :, :);
    history(it) = mean(dc(:).^2); % Sauvegarde de la MSE courante
    
    % --- Étape 7 : Passe arrière (Backward propagation) ---
    % Le gradient d'entrée de la perte quadratique par rapport à la sortie est : 2 * d / N_pixels
    [gW, gb] = srcnn_backward(net, 2 * d / numel(dc), cache);
    
    % --- Étape 8 : Mise à jour des paramètres par l'optimisateur Adam ---
    c1 = 1 - 0.9^it;     % Correction du biais pour le premier moment (beta1 = 0.9)
    c2 = 1 - 0.999^it;   % Correction du biais pour le second moment (beta2 = 0.999)
    
    for l = 1:3
        % Mise à jour des moments pour les poids W
        mW{l} = 0.9 * mW{l} + 0.1 * gW{l};  
        vW{l} = 0.999 * vW{l} + 0.001 * gW{l}.^2;
        % Ajustement des poids W avec régularisation numérique (epsilon = 1e-8)
        net.W{l} = net.W{l} - lr * (mW{l} / c1) ./ (sqrt(vW{l} / c2) + 1e-8);
        
        % Mise à jour des moments pour les biais b
        mb{l} = 0.9 * mb{l} + 0.1 * gb{l};  
        vb{l} = 0.999 * vb{l} + 0.001 * gb{l}.^2;
        % Ajustement des biais b
        net.b{l} = net.b{l} - lr * (mb{l} / c1) ./ (sqrt(vb{l} / c2) + 1e-8);
    end
    
    % --- Étape 9 : Suivi périodique de l'apprentissage (Toutes les 500 itérations) ---
    if mod(it, 500) == 0
        % Évaluation du modèle intermédiaire sur l'image entière du cameraman
        u = srcnn_upscale(camLR, size(camHR), net);
        
        % Calcul du PSNR en base décibel en ignorant 3 pixels de bordures de sécurité
        fprintf('iter %5d | MSE d''entraînement : %.3e | PSNR cameraman : %.2f dB | (Temps : %.0f s)\n', it, ...
            mean(history(it-499:it)), psnr_db(camHR(4:end-3, 4:end-3), ...
            min(max(u(4:end-3, 4:end-3), 0), 1)), toc);
    end
end

% --- Étape 10 : Sauvegarde du modèle final ---
W = net.W; b = net.b; f = net.f;
save('srcnn_trained.mat', 'W', 'b', 'f', 'history');
disp('=== Entraînement terminé. Réseau sauvegardé avec succès dans "srcnn_trained.mat" ===');
