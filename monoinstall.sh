PREFIX=/usr/local
brew install autoconf automake libtool pkg-config cmake
wget "https://download.mono-project.com/sources/mono/mono-4.8.0.524.tar.bz2"
tar xjf mono-4.8.0.524.tar.bz2
cd mono-4.8.0
./configure --prefix=$PREFIX --disable-nls
make
make install
cd $TRAVIS_BUILD_DIR