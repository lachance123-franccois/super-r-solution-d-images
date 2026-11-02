function g = col2im_same(dcols, f, H, W, N, C)
%COL2IM_SAME Opération adjointe/inverse de im2col_same.
%   g = col2im_same(dcols, f, H, W, N, C)
%
%   Entrées :
%       dcols : Matrice 2D de taille (H*W*N) x (f*f*C) contenant les blocs vectorisés.
%       f     : Taille de la fenêtre carrée d'extraction d'origine (f x f).
%       H     : Hauteur cible de l'image.
%       W     : Largeur cible de l'image.
%       N     : Taille du batch (nombre d'images simultanées).
%       C     : Nombre de canaux ou de cartes de caractéristiques.
%
%   Sortie :
%       g     : Matrice accumulée et reformatée de taille (H*W*N) x C.
%
%   Principe :
%       Cette fonction rétropropage ou reconstruit les pixels d'une image en ré-allouant
%       les colonnes de la matrice 2D dans un tenseur 4D. Les pixels des zones de 
%       chevauchement (si f > 1) sont ici sommés (accumulés).

    % Cas optimal : si le filtre est de taille 1x1, aucune opération de voisinage 
    % n'a eu lieu, les données n'ont pas de chevauchements. On renvoie directement la matrice.
    if f == 1
        g = dcols;
        return
    end
    
    % Calcul du rayon du filtre pour gérer le Zero-Padding inverse aux bords
    r = floor(f / 2);
    
    % Allocation d'un tenseur 4D étendu pour accumuler les contributions de chaque bloc
    gp = zeros(H + 2 * r, W + 2 * r, N, C, 'like', dcols);
    
    % Parcours inverse à travers les mêmes décalages géométriques (dy, dx) que im2col_same
    k = 0;
    for dy = 0:f-1
        for dx = 0:f-1
            % Récupération du paquet de C colonnes associé au décalage k et reformatage en bloc 4D
            blk = reshape(dcols(:, k*C + (1:C)), H, W, N, C);
            
            % Accumulation par sommation sur la fenêtre glissante correspondante
            gp(1+dy:H+dy, 1+dx:W+dx, :, :) = gp(1+dy:H+dy, 1+dx:W+dx, :, :) + blk;
            k = k + 1;
        end
    end
    
    % Extraction de la zone utile centrale (on retire le rembourrage de rayon r)
    % et aplatissement au format 2D standard attendu en sortie (H*W*N) x C
    g = reshape(gp(r+1:r+H, r+1:r+W, :, :), H * W * N, C);
end
