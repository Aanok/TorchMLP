# TorchMLP
## Requirements
* [Torch](https://github.com/torch/torch7), single strict requirement for the network on its own.
* [lua-CSV](https://github.com/geoffleyland/lua-csv) to parse the datasets.
* [Torch/gnuplot](https://github.com/torch/gnuplot) and [gnuplot](http://www.gnuplot.info/) for generating graphs during tests

Torch can be installed [headless](http://torch.ch/docs/getting-started.html#_) or from your package manager.

Lua-CSV can be installed from luarocks: `luarocks install csv`.

Torch/gnuplot requires cloning the upstream repository and installing manually by `luarocks make`.

Gnuplot can be installed [headless](http://www.gnuplot.info/download.html) from upstream or from your package manager.
