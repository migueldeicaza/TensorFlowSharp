PREFIX=/usr/local
VERSION=4.8.0.524
wget "https://download.mono-project.com/sources/mono/mono-$VERSION.tar.bz2"
tar xvf mono-$VERSION.tar.bz2
cd mono-$VERSION
./configure --prefix=$PREFIX --disable-nls
make
make install