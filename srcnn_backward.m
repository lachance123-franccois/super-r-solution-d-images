function [gW, gb] = srcnn_backward(net, dout, cache)
%SRCNN_BACKWARD Rétropropagation du SRCNN (gradient de la perte / poids).
%   [gW, gb] = srcnn_backward(net, dout, cache)
%
%   Entrées :
%       net   : Structure contenant les filtres et les poids du réseau.
%       dout  : Tenseur 4D (H x W x 1 x N), dérivée de la perte par rapport à la sortie.
%       cache : Cellule de structures contenant les activations de la passe avant.
%
%   Sorties :
%       gW    : Cellule (3x1) des gradients de la perte par rapport aux poids W.
%       gb    : Cellule (3x1) des gradients de la perte par rapport aux biais b.

    [H, W, ~, N] = size(dout);
    gW = cell(3, 1);
    gb = cell(3, 1);
    
    % Reformatage du gradient d'entrée du format [H, W, 1, N] vers [H*W*N, Cout]
    g = reshape(permute(dout, [1 2 4 3]), H * W * N, []);
    
    % Rétropropagation de la couche 3 vers la couche 1
    for l = 3:-1:1
        c = cache{l};
        
        % Dérivée de l'activation ReLU (uniquement pour les couches 1 et 2)
        if l < 3
            g = g .* (c.out > 0);
        end
        
        % Calcul du gradient par rapport aux poids et aux biais
        gW{l} = c.cols' * g;
        gb{l} = sum(g, 1);
        
        % Propagation du gradient d'erreur vers la couche précédente
        if l > 1
            g = col2im_same(g * net.W{l}', net.f(l), H, W, N, c.cin);
        end
    end
end
