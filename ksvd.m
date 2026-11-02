function D = ksvd(X, K, L, n_iter, verbose)
%   Entrées :
%       X        : Matrice des données d'entraînement de taille d x N (N exemples de dimension d).
%       K        : Nombre d'atomes souhaité dans le dictionnaire cible.
%       L        : Niveau de parcimonie imposé lors de la phase de codage (nb atomes max par signal).
%       n_iter   : Nombre d'itérations globales de l'algorithme K-SVD.
%       verbose  : [Optionnel] Booléen activant l'affichage textuel du suivi de convergence (Défaut: true).
%
%   Sortie :
%       D        : Dictionnaire appris de taille d x K (atomes de norme L2 unitaire).
%
%   Principe (Aharon, Elad & Bruckstein, 2006) :
%       L'algorithme alterne deux étapes principales à chaque itération :
%       (1) Codage parcimonieux global de tous les signaux à l'aide de l'algorithme OMP.
%       (2) Mise à jour itérative, atome par atome, via une SVD économique restreinte.
%           Chaque atome et ses coefficients associés sont recalculés pour minimiser l'erreur
%           quadratique résiduelle, uniquement sur le sous-ensemble de signaux qui l'utilisent.

    % Gestion de l'argument d'affichage par défaut
    if nargin < 5
        verbose = true;
    end
    
    N = size(X, 2); % Nombre total d'exemples disponibles
    
    % --- Initialisation du dictionnaire ---
    % Sélection aléatoire de K signaux de la base d'entraînement pour former les atomes initiaux
    perm = randperm(N);
    D = X(:, perm(1:K));
    
    % Normalisation indispensable des atomes initiaux (Norme L2 égale à 1)
    D = D ./ (sqrt(sum(D.^2, 1)) + 1e-12);
    
    % --- Boucle principale K-SVD ---
    for it = 1:n_iter
        
        % Étape 1 : Codage parcimonieux de l'ensemble de la base sur le dictionnaire actuel
        A = omp(D, X, L);
        
        % Étape 2 : Mise à jour des atomes dans un ordre aléatoire (évite les biais de trajectoire)
        for k = randperm(K)
            
            % Identification des signaux (colonnes) qui utilisent l'atome courant 'k'
            users = find(A(k, :));
            
            % Gestion d'un atome mort/inutilisé
            if isempty(users)
                % On remplace l'atome par l'exemple de la base le plus mal représenté actuellement
                E = X - D * A;
                [~, j] = max(sum(E.^2, 1));
                D(:, k) = X(:, j) / (sqrt(sum(X(:, j).^2)) + 1e-12); % Normalisation sécurisée
                continue
            end
            
            % Élimination temporaire de la contribution de l'atome k pour calculer l'erreur brute
            A(k, users) = 0;
            
            % Calcul de la matrice d'erreur résiduelle restreinte aux signaux utilisateurs
            E = X(:, users) - D * A(:, users);
            
            % Décomposition en Valeurs Singulières économique (SVD rang 1) de la matrice d'erreur
            [U, S, V] = svd(E, 'econ');
            
            % Le premier vecteur propre de U devient le nouvel atome mis à jour (Rang 1 optimal)
            D(:, k) = U(:, 1);
            
            % Réassignation simultanée des coefficients associés ajustés par la valeur singulière S(1,1)
            A(k, users) = S(1, 1) * V(:, 1)';
        end
        
        % Affichage optionnel du Root Mean Squared Error (RMSE) de reconstruction
        if verbose
            fprintf('  K-SVD itération %d : RMSE %.4f\n', it, sqrt(mean(mean((X - D * A).^2))));
        end
    end
end
