#include <array>

using namespace concurrency;
//gradient values (need to make generatable)
float grads[8][2] = { { 1, 1 }, { -1, 1 }, { 1, -1 }, { -1, -1 },
{ 1, 0 }, { -1, 0 }, { 0, -1 }, { 0, 1 } };
float grads2[16] = { 1, 1, -1, 1, 1, -1, -1, -1,
1, 0, -1, 0, 0, -1, 0, 1 };

float grad3[12][3] = { { 1, 1, 0 }, { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 },
{ 1, 0, 1 }, { -1, 0, 1 }, { 1, 0, -1 }, { -1, 0, -1 },
{ 0, 1, 1 }, { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 } };

float grads3[36] = { 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1 };

#define ix(i,j,w,d) ((d)*((i)*(w) + (j)))
#define ixi(xx,w,d) ((xx/d - (xx/d % w))/w)
#define ixj(xx,w,d) (xx/d % w)
#define dot(x1,y1,x2,y2) (x1*x2 + y1*y2)
#define dot33(x1,y1,z1,x2,y2,z2) (x1*x2 + y1*y2 + z1*z2)

//compile time pow function
template<class T>
inline constexpr T constpow(const T base, unsigned const exponent)
{
	// (parentheses not required in next line)
	return (exponent == 0) ? 1 : (base * constpow(base, exponent - 1));
}

// type of data (basic types, float, int etc), and the number of dimensions
template<typename DATATYPE, int nDims>
class NoiseFactory{
	
public:
	//skew constants 2d, 3d, 4d, need to be generated in compile time based on nDims
	//const float F = (sqrt(nDims + 1) - 1) / nDims;
	//const float G = ((nDims + 1) - sqrt(nDims + 1)) / ((nDims + 1) * nDims);
	const int D = constpow(3, nDims)-1;
	int perme[512];
	float gradtemp[3];
	float gradients[(constpow(3, nDims)*nDims)-1];
	const float F = (sqrt(nDims + 1) - 1) / nDims;
	const float G = ((nDims + 1) - sqrt(nDims + 1)) / ((nDims + 1) * nDims);
	void initseeds(){
		int p[256];
		for (int i = 0; i<256; i++){
			p[i] = i;
		}
		int ri, temp;
		for (int i = 0; i<256; i++){
			ri = rand() % 255;
			temp = p[i];
			p[i] = p[ri];
			p[ri] = temp;
		}
		for (int i = 0; i<512; i++){
			perme[i] = p[i & 255];
		//	permMod12[i] = (short)(perm[i] % 12);
		}
		gradtemp[0] = 1;
		gradtemp[1] = -1;
		gradtemp[2] = 0;
		//int N = 0;
		int gradstep[nDims];
		gradstep[0] = 1;
		for (int i = 1; i < nDims; i++){
			gradstep[i] = gradstep[i-1] * 3;
		}
		
		for (int i = 0; i < D; i++){
			int ii = i*nDims;
			for (int d = 0; d < nDims; d++){
				gradients[ii + d] = gradtemp[i / gradstep[d] % 3];
			}
		}
	}
	float OLDnoise2(index<nDims> idx, array_view<float, 1> grad, array_view<int, 1> perm, float F, float G) {
		float o2 = fast_math::pow(2, 1 - 1);
		float xin = idx[0] * 4.0f;
		float yin = idx[1] * 4.0f;
		float zin = idx[2] * 4.0f;
		xin /= 1000.0f;
		yin /= 1000.0f;
		zin /= 1000.0f;
		printf("xins = %f,%f,%f \n", xin, yin, zin);
		float n0 = 0, n1 = 0, n2 = 0, n3 = 0; // Noise contributions from the three corners
		////				 Skew the input space to determine which simplex cell we're in
		float s = (xin + yin + zin)*F; // Very nice and simple skew factor for 3D
		int i = floorf(xin + s);
		int j = floorf(yin + s);
		int k = floorf(zin + s);
		printf("s = %f \n", s);
		printf("ijk = %d,%d,%d \n", i, j, k);
		float G3 = 1.0 / 6.0; // Very nice and simple unskew factor, too
		float t = (i + j + k)*G;
		float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
		float Y0 = j - t;
		float Z0 = k - t;
		float x0 = xin - X0; // The x,y,z distances from the cell origin
		float y0 = yin - Y0;
		float z0 = zin - Z0;
		printf("t = %f \n", t);
		printf("xyz0 = %d,%d,%d \n", x0, y0, z0);
		// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
		// Determine which simplex we are in.
		int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
		int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
		if (x0 >= y0) {
			if (y0 >= z0)
			{
				i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
			} // X Y Z order
			else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; } // X Z Y order
			else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; } // Z X Y order
		}
		else { // x0<y0
			if (y0<z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; } // Z Y X order
			else if (x0<z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; } // Y Z X order
			else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; } // Y X Z order
		}
		// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
		// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
		// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
		// c = 1/6.
		float x1 = x0 - i1 + G; // Offsets for second corner in (x,y,z) coords
		float y1 = y0 - j1 + G;
		float z1 = z0 - k1 + G;
		float x2 = x0 - i2 + 2.0*G; // Offsets for third corner in (x,y,z) coords
		float y2 = y0 - j2 + 2.0*G;
		float z2 = z0 - k2 + 2.0*G;
		float x3 = x0 - 1.0 + 3.0*G; // Offsets for last corner in (x,y,z) coords
		float y3 = y0 - 1.0 + 3.0*G;
		float z3 = z0 - 1.0 + 3.0*G;

		printf("ijk1 = %d,%d,%d \n", i1, j1, k1);
		printf("ijk2 = %d,%d,%d \n", i2, j2, k2);

		printf("xyz1 = %f, %f, %f \n", x1, y1, z1);
		printf("xyz2 = %f, %f, %f \n", x2, y2, z2);
		printf("xyz3 = %f, %f, %f \n", x3, y3, z3);
		// Work out the hashed gradient indices of the four simplex corners
		int ii = i & 255;
		int jj = j & 255;
		int kk = k & 255;
		int gi0 = (perm[ii + perm[jj + perm[kk]]] % 12) * 3;
		int gi1 = (perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12) * 3;
		int gi2 = (perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12) * 3;
		int gi3 = (perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12) * 3;

		printf("gi0123 = %d, %d, %d, %d \n", gi0, gi1, gi2, gi3);
		// Calculate the contribution from the four corners
		float t0 = 0.5 - x0*x0 - y0*y0 - z0*z0;
		float tdot = 0;
		printf("t0= %f \n", t0);
		if (t0<0) n0 = 0.0;
		else {
			t0 *= t0;
			tdot = dot33(grads3[gi0], grads3[gi0 + 1], grads3[gi0 + 2], x0, y0, z0);
			n0 = t0 * t0 * tdot;
		}

		float t1 = 0.5 - x1*x1 - y1*y1 - z1*z1;
		printf("t1= %f \n", t1);
		printf("dot0= %f \n", tdot);
		if (t1<0) n1 = 0.0;
		else {
			t1 *= t1;
			tdot = dot33(grads3[gi1], grads3[gi1 + 1], grads3[gi1 + 2], x1, y1, z1);
			n1 = t1 * t1 * tdot;
		}
		float t2 = 0.5 - x2*x2 - y2*y2 - z2*z2;
		printf("t2= %f \n", t2);
		printf("dot1= %f \n", tdot);
		if (t2<0) n2 = 0.0;
		else {
			t2 *= t2;
			tdot = dot33(grads3[gi2], grads3[gi2 + 1], grads3[gi2 + 2], x2, y2, z2);
			n2 = t2 * t2 * tdot;
		}
		float t3 = 0.5 - x3*x3 - y3*y3 - z3*z3;
		printf("t3= %f \n", t3);
		printf("dot2= %f \n", tdot);
		if (t3<0) n3 = 0.0;
		else {
			t3 *= t3;
			tdot = dot33(grads3[gi3], grads3[gi3 + 1], grads3[gi3 + 2], x3, y3, z3);
			n3 = t3 * t3 * tdot;
		}
		printf("dot3= %f \n", tdot);
		printf("t0123 = %f, %f, %f, %f \n", t0, t1, t2, t3);
		printf("n = %f, %f, %f, %f \n", n0, n1, n2, n3);
		// Add contributions from each corner to get the final noise value.
		// The result is scaled to stay just inside [-1,1]
		printf("v = %f \n \n", 32.0f*(n0 + n1 + n2 + n3) / 1);
		return 32.0f*(n0 + n1 + n2 + n3);
	}
	static float OLDnoise(index<nDims> idx, array_view<float, 1> grad, array_view<int, 1> perm,float F,float G,float o2) restrict(amp,cpu) {
		float xin = idx[0] * o2;
		float yin = idx[1] * o2;
		float zin = idx[2] * o2;
		xin /= 1000.0f;
		yin /= 1000.0f;
		zin /= 1000.0f;
		float n0 = 0, n1 = 0, n2 = 0, n3 = 0; // Noise contributions from the three corners
		////				 Skew the input space to determine which simplex cell we're in
		float s = (xin + yin + zin)*F; // Very nice and simple skew factor for 3D
		int i = (int)fast_math::floor(xin + s);
		int j = (int)fast_math::floor(yin + s);
		int k = (int)fast_math::floor(zin + s);
		float t = (i + j + k)*G;
		float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
		float Y0 = j - t;
		float Z0 = k - t;
		float x0 = xin - X0; // The x,y,z distances from the cell origin
		float y0 = yin - Y0;
		float z0 = zin - Z0;
		// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
		// Determine which simplex we are in.
		int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
		int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
		if (x0 >= y0) {
			if (y0 >= z0)
			{
				i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
			} // X Y Z order
			else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; } // X Z Y order
			else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; } // Z X Y order
		}
		else { // x0<y0
			if (y0<z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; } // Z Y X order
			else if (x0<z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; } // Y Z X order
			else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; } // Y X Z order
		}
		// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
		// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
		// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
		// c = 1/6.
		float x1 = x0 - i1 + G; // Offsets for second corner in (x,y,z) coords
		float y1 = y0 - j1 + G;
		float z1 = z0 - k1 + G;
		float x2 = x0 - i2 + 2.0f*G; // Offsets for third corner in (x,y,z) coords
		float y2 = y0 - j2 + 2.0f*G;
		float z2 = z0 - k2 + 2.0f*G;
		float x3 = x0 - 1.0 + 3.0f*G; // Offsets for last corner in (x,y,z) coords
		float y3 = y0 - 1.0 + 3.0f*G;
		float z3 = z0 - 1.0 + 3.0f*G;
		// Work out the hashed gradient indices of the four simplex corners
		int ii = i & 255;
		int jj = j & 255;
		int kk = k & 255;
		int gi0 = (perm[ii + perm[jj + perm[kk]]] % 12) * 3;
		int gi1 = (perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12) * 3;
		int gi2 = (perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12) * 3;
		int gi3 = (perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12) * 3;
		// Calculate the contribution from the four corners
		float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
		if (t0<0) n0 = 0.0;
		else {
			t0 *= t0;
			n0 = t0 * t0 * dot33(grad[gi0], grad[gi0 + 1], grad[gi0 + 2], x0, y0, z0);
		}
		float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
		if (t1<0) n1 = 0.0;
		else {
			t1 *= t1;
			n1 = t1 * t1 * dot33(grad[gi1], grad[gi1 + 1], grad[gi1 + 2], x1, y1, z1);
		}
		float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
		if (t2<0) n2 = 0.0;
		else {
			t2 *= t2;
			n2 = t2 * t2 * dot33(grad[gi2], grad[gi2 + 1], grad[gi2 + 2], x2, y2, z2);
		}
		float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
		if (t3<0) n3 = 0.0;
		else {
			t3 *= t3;
			n3 = t3 * t3 * dot33(grad[gi3], grad[gi3 + 1], grad[gi3 + 2], x3, y3, z3);
		}
//		 Add contributions from each corner to get the final noise value.
		// The result is scaled to stay just inside [-1,1]
		return 32.0f *(n0 + n1 + n2 + n3);

	}
	// currently an ugly mess of forloops
	static float noise(index<nDims> idx, array_view<float, 1> grad, array_view<int, 1> perm, float F, float G, float o2, int nn, int D2) restrict(amp,cpu) {
		int i;
		float ins[nDims];
		
		for (i = 0; i < nDims; i++){
			ins[i] = (float)((idx[i] * o2) / nn);
		}
		
		float s = 0, t = 0;
		for (i = 0; i < nDims; i++){
			s += ins[i];
		}
		s *= F;
		
		int inds[nDims];
		for (i = 0; i < nDims; i++){
			inds[i] = fast_math::floorf(ins[i] + s);
			t += inds[i];
		}
		t *= G;
		
		float d[nDims + 1][nDims];
		int sorted[nDims];
		//x0
		//HERE!!!
		for (int i = 0; i < nDims; i++){
			d[0][i] = ins[i] - (inds[i] - t);
			sorted[i] = i;
			
		}

		for (int i = 0; i < nDims; i++){
			int l = i;
			for (int j = i + 1; j < nDims; j++){
				if (d[0][sorted[l]] < d[0][sorted[j]]){ l = j; }
			}
			int temp = sorted[i];
			sorted[i] = sorted[l];
			sorted[l] = temp;
		}
		int ij1[nDims - 1][nDims] = { 0 };

		
		for (i = 0; i < nDims - 1; i++){
			for (int j = i; j < nDims - 1; j++){
				ij1[j][sorted[i]] = 1;

			}
		}
		//^^^ SOMETHING HERE!!
		//x1-dims-2
		int gg = 1;
		for (i = 0; i < nDims - 1; i++){
			for (int j = 0; j < nDims; j++){
				d[gg][j] = d[0][j] - ij1[i][j] + (gg*G);
			}
			gg++;
		}
		//x1-dims-1
		for (i = 0; i < nDims; i++){
			d[nDims][i] = d[0][i] - 1 + gg*G;
		}
		int iijj[nDims];
		for (i = 0; i < nDims; i++){
			iijj[i] = inds[i] & 255;
		}
		int gi[nDims + 1];
		float tyt[nDims + 1];
		int g = perm[iijj[nDims - 1]];
		for (i = nDims - 2; i >= 0; i--){
			g = perm[iijj[i] + g];
		}
		gi[0] = (g % D2)*nDims;
		int ttg = 0;
		for (i = 1; i < nDims; i++){
			g = perm[iijj[nDims - 1] + ij1[ttg][nDims - 1]];
			for (int j = nDims - 2; j >= 0; j--){
				g = perm[iijj[j] + ij1[ttg][j] + g];
			}
			ttg++;
			gi[i] = (g % D2)*nDims;
		}
		g = perm[iijj[nDims - 1] + 1];
		for (int j = nDims - 2; j >= 0; j--){
			g = perm[iijj[j] + 1 + g];
		}
		gi[nDims] = (g % D2)*nDims;

		for (i = 0; i < nDims + 1; i++){
			tyt[i] = 0.5f;
			for (int j = 0; j < nDims; j++){
				tyt[i] -= d[i][j] * d[i][j];
			}
		}
		float n[nDims + 1];
		for (i = 0; i < nDims + 1; i++){
			if (tyt[i] < 0){ n[i] = 0.0f; }
			else{
				tyt[i] *= tyt[i];
				float dotf = 0;
				for (int j = 0; j < nDims; j++){
					dotf += grad[gi[i] + j] * d[i][j];
				}
				n[i] = tyt[i] * tyt[i] * dotf;
			}
		}
		float pval = 0;
		for (i = 0; i < nDims + 1; i++){
			pval += n[i];
		}
		pval = (32.0f*pval);
		return pval;
	}
	
	void createnoise(DATATYPE * data, int n,int nocts) {
		int er[3] = { 1000, 1000, 1 };
		int D2 = D;
		extent<nDims> e(er);
		array_view<DATATYPE, nDims> imgar(e, img);
		array_view<int, 1> permi(512, perme);
		array_view<float, 1> gradsi(D*3, gradients);
//skew constants
		float F = (float)(sqrt(nDims + 1) - 1) / nDims;
		float G = (float)((nDims + 1) - sqrt(nDims + 1)) / ((nDims + 1) * nDims);
		parallel_for_each(imgar.extent,	[=](index<nDims> idx) restrict(amp)
		{
			bool istrue = false;
			float val = 1;
			//int ij[nDims];
			//for (int i = 0; i < nDims; i++){
			//	ij[i] = idx[i] / n;
			//}
			for (int o = 1; o <= nocts; o++){
				float o2 = fast_math::powf(2, o - 1);
				val -= noise(idx, gradsi, permi, F, G, o2, n, D2) / o;
			}
		//	float x1 = i1 - 500, y1 = j1 - 500;
		//	float d = fast_math::sqrtf(x1*x1 + y1*y1);

			imgar.get_ref(idx) = fast_math::fabsf(val / nocts);
		});
	}
	//2 dimensional only
	void OLDcreatenoise(DATATYPE * data, int n, int nocts){
		int er[3] = { 1000, 1000, 1 };
		int D2 = 12;
		extent<nDims> e(er);
		array_view<DATATYPE, nDims> imgar(e, img);
		array_view<int, 1> permi(512, perme);
		array_view<float, 1> gradsi(36, grads3);
		float F = (sqrt(nDims + 1) - 1) / nDims;
		float G = ((nDims + 1) - sqrt(nDims + 1)) / ((nDims + 1) * nDims);
		parallel_for_each(
			// Define the compute domain, which is the set of threads that are created.
			imgar.extent,
			// Define the code to run on each thread on the accelerator.
			[=](index<nDims> idx) restrict(amp)
		{
			float val = 1;
			for (int o = 1; o <= nocts; o++){
				float o2 = fast_math::powf(2, o - 1);
				val -= OLDnoise(idx,gradsi,permi,F,G,o2) / o;
			}
			//	float x1 = i1 - 500, y1 = j1 - 500;
			//	float d = fast_math::sqrtf(x1*x1 + y1*y1);
			imgar.get_ref(idx) = fast_math::fabsf(val / nocts);
		});
	}
				
	void noisetest(float * image){
		float F = (sqrt(nDims + 1) - 1) / nDims;
		float G = ((nDims + 1) - sqrt(nDims + 1)) / ((nDims + 1) * nDims);
		
		array_view<int, 1> permi(512, perme);
		array_view<float, 1> gradsi(36, grads3);

		index<3> idx;
		idx[0] = 43;
		idx[1] = 635;
		idx[2] = 0;
		//float ij[nDims];
		for (int i = 0; i < 1000; i++){
			idx[0] = i;
			for (int j = 0; j < 1000; j++){
				idx[1] = j;

				float one = OLDnoise(idx, gradsi, permi, F, G);
				float three = OLDnoise2(idx, gradsi, permi, F, G);
				float two = noise(idx, gradsi, permi, F, G, 4, 1000, 12);
			/*	if (one - two > 0.01f){
					printf("%f, %f,  %d, %d \n", one, two, i, j);
				}*/
				//	image[i * 1000 + j] = one;
			//		printf("one : =  %f,   ", OLDnoise2(idx, gradsi, permi, F, G));
			//	printf("two : =  %f,   ", one);
				//printf("three : =  %f \n", noise(idx, gradsi, permi, F, G, 4, 1000, 12));
			}
		}
	}
	
};