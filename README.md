# Text Classification For Sentiment Analysis

Berikut ini adalah text classification untuk sentiment analysis menggunakan bahasa python. Repository ini dibuat sebagai tugas dari mata kuliah Pengantar Deep learning.

## Requirement
Sebelum menggunakan repository ini, pastikan anda:
1. Telah Menginstall python paling sedikit versi 3.5
2. Menginstall library yang dibutuhkan: pandas, keras, tensorflow, ipython, nltk
3. Pastikan anda memiliki pengetahuan dasar tentang bahasa pemrograman python.

## Instalasi dengan NIX
Nix merupakan tools untuk package management yang digunakan untuk membuat environment sementara seperti pip. Berbeda dengan pip, nix tidak hanya digunakan untuk library python, melainkan seluruh library seperti yang ada di apt milik ubuntu.

Untuk melakukan instalasi Nix pada OS anda, ikuti tutorial pada tautan berikut (ini)[https://nixos.org/nix/download.html]. Apabila Nix telah terinstall, maka jalankan perintah berikut untuk mendownload otomatis library yang diperlukan untuk projek ini:
`nix-build shell.nix`

Untuk memasuki environment sementara, anda dapat mengetikkan perintah:
`nix-shell shell.nix`

Apabila sudah terinstall, maka cuku jalankan ipython notebook sebagai berikut:
`ipython3 notebook`

Dan pilih project yang berekstensikan `.ipynb`

SELAMAT MENCOBA
