%TRAIN_SPARSE_SR Apprend les dictionnaires couplés BR/HR (Zeyde et al., 2010).
%   Ce script orchestre l'extraction des patchs, leur projection par ACP
%   et l'apprentissage alterné des dictionnaires basse et haute résolution.
%
%   Protocole :
%   1. Pour chaque image d'entraînement : HR -> BR (réduction x3) -> MR
%      (BR ré-agrandie en bicubique, même taille que HR).
%   2. Patchs 9x9 (= 3x3 pixels BR) sur une grille de pas 3 :
%        entrée  : caractéristiques de MR (4 dérivées) -> 324 dimensions
%        cible   : détails manquants HR - MR (81 dimensions)
%   3. ACP des caractéristiques (99,9 % de l'énergie conservée, ~25 dimensions).
%   4. Dictionnaire BR Dl appris par K-SVD (512 atomes, 3 atomes par patch).
%   5. Dictionnaire HR Dh par moindres carrés : les mêmes codes A doivent
%      reconstruire les détails HR : Dh = FH * A' * (A * A')^-1.

clear; rng(0); % Fixe le générateur aléatoire pour la reproductibilité

% --- Paramètres globaux ---
scale    = 3;          % Facteur de super-résolution
ps       = 9;          % Taille des patchs carrés (9x9)
step     = 3;          % Pas de déplacement de la grille d'extraction
K        = 512;        % Nombre d'atomes dans le dictionnaire
L        = 3;          % Niveau de parcimonie pour le codage OMP
n_iter   = 20;         % Nombre d'itérations globales pour K-SVD
maxPatch = 40000;      % Nombre maximal de patchs conservés pour l'apprentissage
trainDir = fullfile('data', 'train');

%  Chargement des fichiers 
files = dir(fullfile(trainDir, '*.png','*.jpg'));
nFiles = numel(files);

% Optimisation : Stockage temporaire en cellule pour éviter l'extension dynamique de matrice
cell_FL = cell(1, nFiles);
cell_FH = cell(1, nFiles);

fprintf('--- Extraction des patchs sur %d images ---\n', nFiles);
for f = 1:nFiles
    % 1. Lecture et conversion de l'image haute résolution en niveaux de gris
    hr = to_gray(read_image(fullfile(trainDir, files(f).name)));
    
    % 2. Génération de l'image basse résolution correspondante
    [lr, hr] = degrade(hr, scale);
    
    % 3. Interpolation bicubique initiale (Moyenne Résolution)
    mr = resize_bicubic(lr, size(hr));
    
    % 4. Extraction des caractéristiques locales (gradients) et des hautes fréquences cibles
    fl = feature_patches(mr, ps, step);
    ph = extract_patches(hr - mr, ps, step);
    
    % Stockage temporaire
    cell_FL{f} = fl;
    cell_FH{f} = ph;
end

% Concaténation globale et efficace de tous les patchs extraits
FL = cat(2, cell_FL{:});
FH = cat(2, cell_FH{:});

% --- Nettoyage et sous-échantillonnage de la base de données ---
% On élimine les patchs lisses (sans contraste) qui n'apportent aucune information utile
keep = sum(FL.^2, 1) > (1e-3 * 81);         
FL = FL(:, keep); 
FH = FH(:, keep);

% Limitation du nombre total de patchs pour préserver les performances de K-SVD
if size(FL, 2) > maxPatch
    sel = randperm(size(FL, 2), maxPatch);
    FL = FL(:, sel); 
    FH = FH(:, sel);
end

% Analyse en Composantes Principales (ACP) 
fprintf('Calcul de la réduction de dimension par ACP \n');
% Calcul de la matrice d'autocorrélation et extraction des valeurs/vecteurs propres
[V, E] = eig(FL * FL');
[ev, order] = sort(diag(E), 'descend');
V = V(:, order);

% Recherche de la dimension d minimale permettant de conserver 99.9% de l'énergie
d = find(cumsum(ev) / sum(ev) >= 0.999, 1);
Pm = V(:, 1:d)'; % Matrice de projection ACP [d x 324]

% Application de la réduction de dimension sur les patchs basse résolution
Z = Pm * FL; 
fprintf('Nombre final de patchs : %d | Dimension après ACP : %d (au lieu de %d)\n', size(FL, 2), d, size(FL, 1));

%  Apprentissage des Dictionnaires 
fprintf(' Début de l''apprentissage K-SVD \n');
tic;

% 1. Apprentissage du dictionnaire Basse Résolution Dl dans l'espace réduit par ACP
Dl = ksvd(Z, K, L, n_iter);

% 2. Calcul final des codes de projection parcimonieux avec OMP
A = omp(Dl, Z, L);

% 3. Calcul analytique du dictionnaire Haute Résolution Dh par moindres carrés
% Équivalent à la formule pseudo-inverse : Dh = FH * A' * inv(A * A')
% On ajoute une petite régularisation pour stabiliser l'inversion numérique
Dh = (FH * A') / (A * A' + 1e-6 * eye(size(A, 1)));

fprintf('Apprentissage des dictionnaires termine en %.1f minutes.\n', toc / 60);

% Sauvegarde du modèle 
save('sparse_sr_model.mat', 'Pm', 'Dl', 'Dh', 'L', 'ps', 'scale');
fprintf('Modele enregistre avec succes dans "sparse_sr_model.mat"\n');
