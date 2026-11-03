function save_figure(fig, filename)
%SAVE_FIGURE Enregistre une figure en PNG (crée le dossier si besoin).
    folder = fileparts(filename);
    if ~isempty(folder) && ~exist(folder, 'dir')
        mkdir(folder);
    end
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, filename, '-dpng', '-r130');
end
