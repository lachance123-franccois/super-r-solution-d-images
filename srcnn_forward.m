function [out, cache] = srcnn_forward(net, x)
%SRCNN_FORWARD Passe avant du réseau SRCNN (Variante Résiduelle).
%   [out, cache] = srcnn_forward(net, x)
%
%   Entrées :
%       net   : Structure contenant les paramètres du réseau :
%                 - net.f : Vecteur des tailles de filtres [9, 1, 5]
%                 - net.W : Cellule des matrices de poids pour les 3 couches
%                 - net.b : Cellule des vecteurs de biais pour les 3 couches
%       x     : Tenseur 4D de l'image d'entrée de taille H x W x 1 x N.
%               (Image basse résolution pré-agrandie en bicubique, normalisée dans [0, 1]).
%
%   Sorties :
%       out   : Tenseur 4D de l'image super-résolue de taille H x W x 1 x N.
%       cache : [Optionnel] Cellule de structures stockant les activations intermédiaires.
%               (Indispensable pour la rétropropagation lors de l'entraînement).
%
%   Architecture (Dong et al., 2014 - Version Résiduelle) :
%       Couche 1 : Convolution 9x9, 64 filtres + activation ReLU (Extraction de patchs)
%       Couche 2 : Convolution 1x1, 32 filtres + activation ReLU (Cartographie non-linéaire)
%       Couche 3 : Convolution 5x5, 1 filtre sans activation    (Reconstruction du résidu)

    [H, W, ~, N] = size(x);
    keep = nargout > 1;       % Vérifie si les variables de cache sont demandées (mode entraînement)
    cache = cell(3, 1);       % Allocation de la cellule de cache
    h = x;                    % Initialisation de la variable de flux avec l'image d'entrée
    
    % Parcours des 3 couches de convolution 
    for l = 1:3
        % 1. Transformation du tenseur d'image 4D en matrice géante de voisinages
        cols = im2col_same(h, net.f(l));
        
        % 2. Calcul de la convolution par produit matriciel performant et ajout du biais
        o = cols * net.W{l} + net.b{l};
        
        % 3. Application de la fonction d'activation non-linéaire ReLU (uniquement couches 1 et 2)
        if l < 3
            o = max(o, 0);
        end
        
        % 4. Sauvegarde des données intermédiaires si le mode entraînement est actif
        if keep
            cache{l} = struct('cols', cols, 'out', o, 'cin', size(h, 3));
        end
        
        % 5. Reformatage de la matrice 2D résultante pour redonner une forme de tenseur d'image 4D.
        % On redimensionne d'abord en [H, W, N, Cout] puis on permute pour retrouver l'ordre [H, W, Cout, N]
        h = permute(reshape(o, H, W, N, size(net.W{l}, 2)), [1 2 4 3]);
    end
    
    % Étape finale de l'architecture résiduelle : 
    % L'image finale est la somme de l'image interpolée d'origine (x) et du résidu de détails (h)
    out = x + h;
end
