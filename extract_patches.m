function [P, ii, jj] = extract_patches(img, ps, step)
%EXTRACT_PATCHES Extraction de patchs ps x ps sur une grille régulière.
%   [P, ii, jj] = extract_patches(img, ps, step)
%
%   Entrées :
%       img  : Image source en niveaux de gris (matrice 2D de taille H x W).
%       ps   : Taille d'un côté du patch carré (ps x ps).
%       step : Pas de déplacement de la grille d'extraction (strides).
%
%   Sorties :
%       P    : Matrice de taille (ps^2) x N contenant les patchs vectorisés par colonne.
%       ii   : Vecteur colonne (N x 1) des indices de ligne du coin haut-gauche de chaque patch.
%       jj   : Vecteur colonne (N x 1) des indices de colonne du coin haut-gauche de chaque patch.
%
%   Principe :
%       La fonction échantillonne l'image à pas régulier et force l'inclusion 
%       systématique des frontières géométriques inférieures et droites (bords inclus).

    [H, W] = size(img);
    
    % Détermination des coordonnées de départ des lignes (axes verticaux)
    ri = 1:step:H-ps+1;
    if ri(end) ~= H - ps + 1
        ri(end+1) = H - ps + 1; % Inclusion forcée de la bordure inférieure
    end
    
    % Détermination des coordonnées de départ des colonnes (axes horizontaux)
    ci = 1:step:W-ps+1;
    if ci(end) ~= W - ps + 1
        ci(end+1) = W - ps + 1; % Inclusion forcée de la bordure droite
    end
    
    % Génération rapide de la grille bidimensionnelle sans boucle (coordonnées globales)
    [I, J] = ndgrid(ri, ci); 
    ii = I(:); 
    jj = J(:);
    N = numel(ii); % Nombre total de patchs collectés
    
    % Allocation de la matrice de stockage des patchs vectorisés
    P = zeros(ps * ps, N);
    
    % Extraction vectorisée : on parcourt les pixels relatifs à l'intérieur du patch
    k = 0;
    for b = 0:ps-1                           % Décalage de colonne dans le patch
        for a = 0:ps-1                       % Décalage de ligne dans le patch
            k = k + 1;
            % Conversion des coordonnées 2D en indices linéaires pour un accès mémoire ultra-rapide
            P(k, :) = img(sub2ind([H, W], ii + a, jj + b))';
        end
    end
end
