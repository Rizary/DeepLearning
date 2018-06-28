{ nixpkgs ? import ./nixdeps/nixpkgs.nix }:
let
   pkgs = import nixpkgs {};
   env = (pkgs.python3.withPackages
     (pythonPackages: with pythonPackages; [
       ipykernel
       pandas
       scikitlearn
       nltk
       tensorflow
       numpy
       scipy
       matplotlib
       tqdm
       jupyter
       gensim
       scrapy
       termcolor
       Keras
       punkt
       ]));
  
in pkgs.stdenv.mkDerivation rec {
     name = "qoeifs-app";
     src = ./.;
     installPhase = ''
       mkdir -p $out/src
       cp -rf $src/* $out/src
     '';
     phases = ["unpackPhase" "installPhase" ];
     buildInputs = [ env ];
   }

