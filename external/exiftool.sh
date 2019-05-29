wget https://www.sno.phy.queensu.ca/~phil/exiftool/Image-ExifTool-11.44.tar.gz
gzip -dc Image-ExifTool-11.44.tar.gz | tar -xf -
cd ./Image-ExifTool-11.44
perl Makefile.PL
make test
sudo make install