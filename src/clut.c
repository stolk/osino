
#include <math.h>

inline float lin2srgb(float val)
{
	if (val <= 0.0031308f)
		return 12.92f * val;
	else
		return 1.055f * powf(val, 1.0f / 2.4f) - 0.055f;
}


inline float srgb2lin(float val)
{
	if (val < 0.04045f)
		return val * (1.0f / 12.92f);
	else
		return powf((val + 0.055f) * (1.0f / 1.055f), 2.4f);
}


float clut[256][3];


static void clut_add_entry(int i, int col)
{
	float r = ((col>>16)&0xff) / 255.0f;
	float g = ((col>> 8)&0xff) / 255.0f;
	float b = ((col>> 0)&0xff) / 255.0f;
	// make linear
	clut[i][0] = srgb2lin(r);
	clut[i][1] = srgb2lin(g);
	clut[i][2] = srgb2lin(b);
}


void clut_init(int palnr, int shft)
{
	int palette[7][4] =
	{
		0xEA6283, 0xF8F335, 0x99D9DA, 0x000000,
		0xF09D00, 0x55A5FF, 0x22259F, 0xFFFFFF,
		0xE66C4F, 0x64DB8F, 0x55A5FF, 0xFFFFFF,
		0x791FFF, 0xF6D200, 0x64DB8F, 0x000000,
		0xF9F572, 0x3C449F, 0xEA6162, 0x000000,
		0x0C163D, 0xE26B00, 0xF4C500, 0xFFFFFF,
		0x7FC2C2, 0xE1774B, 0xF4DB60, 0xFFFFFF,
	};
	const int numpal = sizeof(palette) / (4*sizeof(int));
	if (palnr<0)
		palnr = numpal-1;
	palnr = palnr % numpal;

	const int col0 = palette[palnr][(0+shft)%4];
	const int col1 = palette[palnr][(1+shft)%4];
	const int col2 = palette[palnr][(2+shft)%4];
	const int col3 = palette[palnr][(3+shft)%4];

	const int stripes[14] =
	{
		col3,
		col3,
		col1,
		col1,
		col3,
		col3,
		col2,
		col2,
		col3,
		col3,
		col1,
		col1,
		col3,
		col3,
	};

	for (int i=0; i<256; ++i)
		clut_add_entry(i, col0);

	for (int i=0; i<14; ++i)
		clut_add_entry(120+i, stripes[i]);
}

