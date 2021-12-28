#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CImg.h"
#include <stdio.h>
#include <cstring>
#include <string>
#include "kernels.h"

#include <cstdlib>
#include <cuda.h>
#include <functional>

#define DIPLAY_SIZE 400

using namespace std;
using namespace cimg_library;

int main(int argc,const char** argv)
{
	// Sprawdzane czy podano wymagane argumenty
	if (argc != 3) {
		cout << "Podaj nazwę pliku wejściowego i wyjściowego!" << endl;
		return 0;
	}

	// TYLKO DO TESTÓW, USUNAC POTEM
	// if(argc != 5) {
	//	 cout << "Podaj rozmiary resize" << endl;
	//	 return 0;
	// }
	// long szer = strtol(argv[3], NULL, 10);
	// long wys = strtol(argv[4], NULL, 10);

	// Wczytanie argumentów
	const char* input_file = argv[1];
	const char* output_file = argv[2];

	// Podanie ścieżki do biblioteki magick, aby wczytać jpg
	// cimg::imagemagick_path("D:\\CUDAprograms\\ImageMagick-7.0.9-Q16\\magick.exe");
	// Wczytanie obrazka
	// CImg<unsigned char> src(input_file);

	// Wczytanie obrazu oraz tworzenie pomocniczych obrazów
	CImg<unsigned char> image(input_file),
		  src(image.width(), image.height(), 1, 1, 0),
		  imgR(image.width(), image.height(), 1, 3, 0),
		  imgG(image.width(), image.height(), 1, 3, 0),
		  imgB(image.width(), image.height(), 1, 3, 0);
 
	// Dla każdego piksela obrazu
	cimg_forXY(image,x,y) {
		imgR(x,y,0,0) = image(x,y,0,0),	// Czerwony kanał
		imgG(x,y,0,1) = image(x,y,0,1),	// Zielony kanał
		imgB(x,y,0,2) = image(x,y,0,2);	// Niebieski kanał
	
		// Separacja kanałów
		int R = (int)image(x,y,0,0);
		int G = (int)image(x,y,0,1);
		int B = (int)image(x,y,0,2);
		// Szary to średnia ważona kanałów
		int grayValueWeight = (int)(0.299*R + 0.587*G + 0.114*B);
		// zapisanie szarego
		src(x,y,0,0) = grayValueWeight;
	}

	// TODO: Pozbyć się tego resize na sztywno
	// src.resize(szer, wys);
	src.resize(512, 512);
	// Zapisanie szerokosci i wysokosci obrazka
	int szerokosc = src.width();
	int wysokosc = src.height();
	unsigned long int rozmiar = src.size();

	// TYLKO DO TESTÓW, USUNAC POTEM
	// cout << "szerokosc" << szerokosc << "wysokosc" << wysokosc << "rozmiar" << rozmiar << endl;

	// Stworzenie wskaźników na piksele na host 
	unsigned char *h_wsk = src.data();
	// unsigned char *h_wsk2 = src.data();
	CImg<unsigned char>bin(szerokosc, wysokosc, 1, 1);
	unsigned char *h_bin = bin.data();

	filter(h_wsk, h_bin, szerokosc, wysokosc, rozmiar);

	// bin.normalize(0, 255);
	bin.save(output_file);

	// Resize tylko po to aby zmieściło się na ekranie
	image.resize(DIPLAY_SIZE, DIPLAY_SIZE);
	bin.resize(DIPLAY_SIZE, DIPLAY_SIZE);

	CImgDisplay main_disp2(image, "Obraz wejsciowy");
	CImgDisplay display(bin, "Obraz wyjsciowy");
	while (!display.is_closed() && main_disp2.is_closed())
	{
		display.wait();
		main_disp2.wait();
	}

	cout << "Naciśnij 'Enter' aby zakończyć";
	cin.get();

	return 0;
}