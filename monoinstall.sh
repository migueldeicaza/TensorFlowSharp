PATH=$PREFIX/bin:$PATH
git clone https://github.com/mono/mono.git
cd mono
CC='cc -m32' ./autogen.sh --prefix=$PREFIX --disable-nls
make
make install