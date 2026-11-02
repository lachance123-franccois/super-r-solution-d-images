function F = sr_features(mr)
%   Entrée :
%       mr      : Image interpolée (Moyenne Résolution) de taille H x W (niveaux de gris).
%
%   Sortie :
%       F       : Tenseur 3D de taille H x W x 4 contenant 4 cartes de caractéristiques.
%                 (Dérivées premières et secondes, horizontales et verticales).
%
%   Principe :
%       Ces filtres décrivent de manière robuste la structure locale (contours, textures)
%       en s'affranchissant de la luminance moyenne (composante continue / DC). Cela rend 
%       l'apprentissage par dictionnaire ou le codage parcimonieux beaucoup plus efficace.

    % Définition des filtres de gradients (standards de la littérature Yang 2010, Zeyde 2010)
    f1 = [-1 0 1];                    % Dérivée première horizontale (extraction de contours verticaux)
    f2 = [1 0 -2 0 1] / 2;            % Dérivée seconde horizontale (extraction des hautes fréquences)
    filters = {f1, f1', f2, f2'};     % Regroupement en cellule : {1er_H, 1er_V, 2e_H, 2e_V}
    
    [H, W] = size(mr);
    F = zeros(H, W, 4);               % Allocation du tenseur de caractéristiques de sortie
    
    % Rayon maximal de rembourrage (ici r=2 car le filtre de taille maximale fait 5 pixels)
    r = 2;
    
    % Indirection d'indices pour générer une extension miroir/symétrique aux frontières
    ri = [r:-1:1, 1:H, H:-1:H-r+1];          
    ci = [r:-1:1, 1:W, W:-1:W-r+1];
    P = mr(ri, ci);                   % Image rembourree de manière symétrique aux bords
    
    % Filtrage et extraction des caractéristiques
    for k = 1:4
        % Convolution 2D sur l'image étendue
        G = conv2(P, filters{k}, 'same');
        
        % Extraction de la zone centrale utile correspondant exactement à la géométrie de mr
        F(:, :, k) = G(r+1:r+H, r+1:r+W);
    end
end
