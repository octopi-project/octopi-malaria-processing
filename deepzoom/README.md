## Requirements
### Mac
```
brew install vips
pip3 install pyvips
```
### Ubuntu
The script didn't work on a Ubuntu 18.04.5 computer - problem with libvips. The script works on Ubuntu 20.04.3
```
sudo apt-get install libvips-dev
pip3 install pyvips
```

### Running the script on Ubuntu
Need to set the number of files 
```
ulimit -n 4096
```
