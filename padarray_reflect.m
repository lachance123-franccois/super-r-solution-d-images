function P = padarray_reflect(f, r)
%   Entrées :
%       f : Image source (matrice 2D en niveaux de gris ou tenseur 3D couleur/multicanal).
%       r : Nombre de pixels à ajouter pour le rembourrage (le rayon r doit être < taille de l'image).
%
%   Sortie :
%       P : Image étendue de taille (H+2r) x (W+2r) x C.
%
%   Exemple pour r = 2 :
%       Une ligne d'origine [a b c d] devient la séquence étendue : [c b | a b c d | c b].
%       Le pixel de bordure ('a' ou 'd') n'est pas dupliqué de façon contiguë, ce qui préserve
%       la continuité des gradients locaux aux frontières de l'image.

    [H, W, ~] = size(f);
    
    % Construction des vecteurs d'indirection pour les lignes (ri) et les colonnes (ci)
    % r+1:-1:2  -> Reflet miroir avant le premier élément (sans répéter l'élément 1)
    % 1:H       -> Conservation des indices de l'image d'origine au centre
    % H-1:-1:H-r -> Reflet miroir après le dernier élément (sans répéter l'élément H)
    ri = [r+1:-1:2, 1:H, H-1:-1:H-r];
    ci = [r+1:-1:2, 1:W, W-1:-1:W-r];
    
    % Application de l'indirection sur l'ensemble des canaux (:) de manière ultra-rapide
    P = f(ri, ci, :);
end
