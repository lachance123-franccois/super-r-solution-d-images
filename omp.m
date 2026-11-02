function A = omp(D, X, L)
%OMP Orthogonal Matching Pursuit (Poursuite Orthogonale), un signal par colonne.
%   A = omp(D, X, L)
%
%   Entrées :
%       D : Dictionnaire de taille d x K (K atomes vectorisés de dimension d, de norme 1).
%       X : Matrice des signaux de taille d x N (N exemples/patchs à encoder).
%       L : Seuil de parcimonie (nombre maximal d'atomes autorisés par signal).
%
%   Sortie :
%       A : Matrice des codes de taille K x N (au plus L coefficients non nuls par colonne).
%
%   Principe :
%       Pour chaque signal, l'algorithme sélectionne itérativement l'atome le plus
%       corrélé au résidu actuel. Il met ensuite à jour l'ensemble des coefficients
%       des atomes déjà sélectionnés en résolvant un problème de moindres carrés.

    K = size(D, 2); % Nombre total d'atomes dans le dictionnaire
    N = size(X, 2); % Nombre de signaux à traiter
    A = zeros(K, N); % Allocation de la matrice de sortie des codes parcimonieux
    
    % Traitement individuel de chaque signal (colonne par colonne)
    for n = 1:N
        x = X(:, n); % Signal courant
        r = x;       % Initialisation du résidu avec le signal d'origine
        
        % Optimisation : Pré-allocation des indices pour éviter l'extension dynamique
        idx = zeros(1, L); 
        a = [];
        
        for s = 1:L
            % 1. Calcul des projections (corrélations) entre le dictionnaire et le résidu
            c = abs(D' * r);
            
            % En désactivant les atomes déjà sélectionnés lors des étapes précédentes
            if s > 1
                c(idx(1:s-1)) = 0;
            end
            
            % 2. Sélection de l'atome présentant la plus forte corrélation
            [~, best] = max(c);
            idx(s) = best; 
            
            % 3. Extraction des sous-atomes sélectionnés pour former la base courante
            Ds = D(:, idx(1:s));
            
            % 4. Résolution du problème des moindres carrés par division matricielle (backslash)
            % Équivalent à projeter orthogonalement le signal d'origine x sur l'espace de Ds
            a = Ds \ x;
            
            % 5. Mise à jour du résidu (composante du signal non encore capturée)
            r = x - Ds * a;
            
            % Critère d'arrêt précoce : si le résidu est orthogonalement nul (erreur négligeable)
            if (r' * r) < 1e-12
                idx = idx(1:s); % Ajustement de la taille utile de l'index
                break
            end
        end
        
        % Stockage des coefficients calculés à leurs positions d'atomes respectives
        A(idx, n) = a;
    end
end
