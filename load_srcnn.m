function net = load_srcnn(filename)
    s = load(filename);
    net.f = double(s.f(:)');
    net.W = s.W(:)';
    net.b = s.b(:)';
    for l = 1:3
        net.W{l} = double(net.W{l});
        net.b{l} = double(net.b{l}(:)');
    end
end
