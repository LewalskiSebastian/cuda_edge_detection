#include "kernels.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CImg.h"
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>

const int DIPLAY_SIZE = 400;

using namespace std;
using namespace cimg_library;

int main(int argc, const char** argv) {
	// Sprawdzane czy podano wymagane argumenty
	if (argc != 3) {
		cout << "Podaj nazwę pliku wejściowego i wyjściowego!" << endl;
		return 0;
	}

	// Wczytanie argumentów
	const char* inputFile = argv[1];
	const char* outputFile = argv[2];

	// Podanie ścieżki do biblioteki magick, aby wczytać jpg
	// cimg::imagemagick_path("D:\\CUDAprograms\\ImageMagick-7.0.9-Q16\\magick.exe");

	// Wczytanie obrazu oraz tworzenie pomocniczych obrazów
	CImg<unsigned char> image(inputFile),
		  src(image.width(), image.height(), 1, 1, 0),
		  imgR(image.width(), image.height(), 1, 3, 0),
		  imgG(image.width(), image.height(), 1, 3, 0),
		  imgB(image.width(), image.height(), 1, 3, 0);
 
	// Dla każdego piksela obrazu
	cimg_forXY(image, x, y) {
		imgR(x, y, 0, 0) = image(x, y, 0, 0),	// Czerwony kanał
		imgG(x, y, 0, 1) = image(x, y, 0, 1),	// Zielony kanał
		imgB(x, y, 0, 2) = image(x, y, 0, 2);	// Niebieski kanał

		// Separacja kanałów
		int r = static_cast<int>(image(x, y, 0, 0));
		int g = static_cast<int>(image(x, y, 0, 1));
		int b = static_cast<int>(image(x, y, 0, 2));
		// Szary to średnia ważona kanałów
		int grayValueWeight = static_cast<int>(0.299*r + 0.587*g + 0.114*b);
		// zapisanie szarego
		src(x, y, 0, 0) = grayValueWeight;
	}

	// TODO: Pozbyć się tego resize na sztywno
	// src.resize(szer, wys);
	src.resize(512, 512);
	// Zapisanie szerokosci i wysokosci obrazka
	int width = src.width();
	int height = src.height();
	unsigned long int size = src.size();

	// Stworzenie wskaźników na piksele na host
	unsigned char *h_ptr = src.data();
	// unsigned char *h_ptr2 = src.data();
	CImg<unsigned char> bin(width, height, 1, 1);
	unsigned char *h_bin = bin.data();

	filter(h_ptr, h_bin, width, height, size);

	// bin.normalize(0, 255);
	bin.save(outputFile);

	// Resize tylko po to aby zmieściło się na ekranie
	image.resize(DIPLAY_SIZE, DIPLAY_SIZE);
	bin.resize(DIPLAY_SIZE, DIPLAY_SIZE);

	CImgDisplay inputDisp(image, "Obraz wejsciowy");
	CImgDisplay outputDisp(bin, "Obraz wyjsciowy");
	while (!outputDisp.is_closed() && inputDisp.is_closed()) {
		outputDisp.wait();
		inputDisp.wait();
	}

	cout << "Naciśnij 'Enter' aby zakończyć";
	cin.get();

	return 0;
}