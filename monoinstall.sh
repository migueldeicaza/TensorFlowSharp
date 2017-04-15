PREFIX=/usr/local
VERSION=4.8.1.0
tar xvf mono-$VERSION.tar.bz2
cd mono-$VERSION
./configure --prefix=$PREFIX --disable-nls
make
make install