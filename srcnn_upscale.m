function hr = srcnn_upscale(lr, outSize, net)
%   Entrées :
%       lr      : Image basse résolution (matrice 2D en niveaux de gris).
%       outSize : Taille cible [H, W] attendue pour l'image haute résolution.
%       net     : Structure contenant les paramètres et poids du réseau SRCNN.
%
%   Sortie :
%       hr      : Image haute résolution finale estimée et corrigée des effets de bord.
%
%   Principe :
%       L'image basse résolution subit un pré-agrandissement bicubique à la taille cible (MR).
%       Elle est ensuite étendue par symétrie miroir de 6 pixels avant d'être corrigée par 
%       le réseau. Ce rembourrage absorbe le rayon du champ réceptif global (4 + 0 + 2) et 
%       évite les artefacts noirs ou flous sur les contours externes lors des convolutions.

    % Étape 1 : Pré-agrandissement géométrique à la taille haute résolution cible
    mr = imresize(lr, outSize,'bicubic');
    
    % Étape 2 : Extension symétrique miroir pour protéger les frontières de l'image
    r = 6;
    x = padarray_reflect(mr, r);
    
    % Étape 3 : Passe avant à travers les trois couches convolutives du SRCNN
    y = srcnn_forward(net, x);
    
    % Étape 4 : Dé-rembourrage pour extraire uniquement la zone centrale utile nette
    hr = y(r+1:end-r, r+1:end-r);
end
