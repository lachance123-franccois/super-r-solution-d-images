function hr = sparse_sr_upscale(lr, outSize, model, step)
%SPARSE_SR_UPSCALE Super-résolution par représentations parcimonieuses.
%   hr = sparse_sr_upscale(lr, outSize, model)
%   hr = sparse_sr_upscale(lr, outSize, model, step)
%
%   Entrées :
%       lr      : Image basse résolution (matrice 2D en niveaux de gris).
%       outSize : Taille cible [H, W] attendue pour l'image haute résolution.
%       model   : Structure contenant les matrices Pm (ACP), Dl et Dh (dictionnaires),
%                 ainsi que les hyperparamètres L (parcimonie) et ps (taille des patchs).
%       step    : [Optionnel] Pas des patchs pour la reconstruction (Défaut : 1 pour un max de détails).
%
%   Sortie :
%       hr      : Image haute résolution finale estimée.
%
%   Algorithme :
%       1. Pré-agrandissement de l'image basse résolution par interpolation bicubique (MR).
%       2. Extraction et projection ACP des patchs de caractéristiques (gradients).
%       3. Codage parcimonieux à l'aide de OMP sur le dictionnaire basse résolution Dl.
%       4. Génération des détails haute fréquence correspondants via le dictionnaire Dh.
%       5. Fusion et moyenne des blocs qui se chevauchent, puis sommation avec l'image MR.

    % Gestion du pas d'échantillonnage par défaut (1 pixel assure la meilleure qualité)
    if nargin < 4
        step = 1;
    end
    
    ps = model.ps; % Récupération de la taille des patchs (ex: 9)
    
    % Étape 1 : Agrandissement géométrique initial pour créer l'image Moyenne Résolution
    mr = resize_bicubic(lr, outSize);
    
    % Étape 2 : Extraction des patchs de caractéristiques (gradients locaux d'ordre 1 et 2)
    [FL, ii, jj] = feature_patches(mr, ps, step);
    
    % Étape 3 : Projection ACP des caractéristiques et calcul des codes parcimonieux
    % La matrice de projection model.Pm réduit la dimension avant l'exécution de l'OMP
    A = omp(model.Dl, model.Pm * FL, model.L);
    
    % Étape 4 : Reconstruction linéaire des détails haute résolution (Hautes Fréquences)
    PH = model.Dh * A;
    
    % Étape 5 : Grille d'accumulation Overlap-Add (Somme et moyenne des blocs chevauchants)
    acc = zeros(size(mr)); % Matrice d'accumulation des pixels haute fréquence
    cnt = zeros(size(mr)); % Matrice de comptage (compteur de vote par pixel)
    
    % Reconstruction bloc par bloc
    for k = 1:numel(ii)
        % Détermination de la fenêtre d'indices 2D associée au patch courant
        r = ii(k) : ii(k) + ps - 1;
        c = jj(k) : jj(k) + ps - 1;
        
        % Dépliage (reshape) du vecteur colonne de détails en bloc carré et accumulation
        acc(r, c) = acc(r, c) + reshape(PH(:, k), ps, ps);
        
        % Mise à jour du compteur de chevauchement
        cnt(r, c) = cnt(r, c) + 1;
    end
    
    % Calcul de la moyenne des détails (pondération par le nombre de blocs par pixel)
    % et combinaison avec l'image interpolée de base (MR) pour produire l'image HR finale
    hr = mr + acc ./ max(cnt, 1);
end
