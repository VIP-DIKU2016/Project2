Software needed:
    python 2.7, openCV, scipy, numpy

    We need a build of openCV that supports SIFT (although most likely this is not the right descriptor algorithms to use, see code comments).
    Did not manage to install it using pip, only with brew:
    `$ brew install opencv3 --force --with-contrib --HEAD`
    `$ echo /usr/local/opt/opencv3/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/opencv3.pth # will be displayed by brew after install`

Correct usage: 
    python code.py <method>
Methods available: 
    1 - Blob detection  
    2 - Harris corner detection
    3 - SIFT feature matching

Examples:
    python code.py 1
