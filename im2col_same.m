function cols = im2col_same(h, f)
%IM2COL_SAME Voisinages f x f pour une convolution « same » (bord : zéros).
%   cols = im2col_same(h, f)
%
%   Entrée :
%       h    : Tenseur 4D de taille H x W x C x N (Hauteur, Largeur, Canaux, Batch)
%       f    : Taille de la fenêtre carrée de convolution (f x f)
%
%   Sortie :
%       cols : Matrice 2D de taille (H*W*N) x (f*f*C) où chaque ligne contient
%              un voisinage vectorisé. Bloc k = f*dy + dx de C colonnes.
%
%   Principe :
%       La convolution glissante à travers le tenseur se transforme en un unique
%       produit matriciel performant : cols * W (où W représente les poids du réseau).

    [H, W, C, N] = size(h);
    
    % Cas optimal : si le filtre est de taille 1x1, pas besoin de voisinage glissant.
    % On réorganise directement les dimensions du tenseur par permutation.
    if f == 1
        cols = reshape(permute(h, [1 2 4 3]), H * W * N, C);
        return
    end
    
    % Calcul du rayon du filtre pour le Zero-Padding
    r = floor(f / 2);
    
    % Remplissage par des zéros (Same Padding) pour préserver la taille H x W en sortie
    hp = zeros(H + 2 * r, W + 2 * r, C, N, 'like', h);
    hp(r+1:r+H, r+1:r+W, :, :) = h; % Insertion de l'image d'origine au centre
    
    % Allocation de la matrice finale de voisinages
    cols = zeros(H * W * N, f * f * C, 'like', h);
    
    % Extraction des blocs par décalages successifs (dy, dx)
    k = 0;
    for dy = 0:f-1
        for dx = 0:f-1
            % Isolement de la fenêtre décalée et réorganisation des dimensions [H, W, N, C]
            blk = permute(hp(1+dy:H+dy, 1+dx:W+dx, :, :), [1 2 4 3]);
            
            % Vectorisation et assignation par paquets de C colonnes dans la matrice de sortie
            cols(:, k*C + (1:C)) = reshape(blk, H * W * N, C);
            k = k + 1;
        end
    end
end
