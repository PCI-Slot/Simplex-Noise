#include <array>

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

// type of data (basic types, float, int etc), and the number of dimensions
template<typename DATATYPE, int nDims>
class NoiseFactory{
public:
	//skew constants 2d, 3d, 4d, need to be generated in compile time based on nDims
	const float F2 = 0.5*(sqrt(3.0) - 1.0);
	const float G2 = (3.0 - sqrt(3.0)) / 6.0;
	const float F3 = 1.0 / 3.0;
	const float G3 = 1.0 / 6.0;
	const float F4 = (sqrt(5.0) - 1.0) / 4.0;
	const float G4 = (5.0 - sqrt(5.0)) / 20.0;
	const int nGrads = 20;
	float gradients[nDims*nDims];
	int perm[512];
	float gradtemp[3];
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
			perm[i] = p[i & 255];
		//	permMod12[i] = (short)(perm[i] % 12);
		}
		gradtemp[0] = 1;
		gradtemp[1] = -1;
		gradtemp[2] = 0;
		int N = 0;
		//for (int i = 0; i < 3; i++){
		//	int ii = i*nDims;
		//	for (int d = 0; d < 3; d++){
		//		gradients[ii+d] = 
		//	}
		//}
	}
	void createnoise(DATATYPE * data, int n,int nocts){
		array_view<DATATYPE, nDims> imgar(n, n, img);
		array_view<int, 1> permi(512, perm);
		array_view<float, 1> gradsi(36, grads3);
		float F2 = 0.5*(sqrt(3.0) - 1.0);
		float G2 = (3.0 - sqrt(3.0)) / 6.0;
		parallel_for_each(
			// Define the compute domain, which is the set of threads that are created.
			imgar.extent,
			// Define the code to run on each thread on the accelerator.
			[=](index<nDims> idx) restrict(amp)
		{
			float val = 1;
			float ij[nDims];
			for (int i = 0; i < nDims; i++){
				ij[i] = idx[i];
			}
			for (int o = 1; o <= nocts; o++){
				float o2 = fast_math::pow(2, o - 1);
				float ins[nDims];
				for (int i = 0; i < nDims; i++){
					ins[i] = (ij[i] * o2) / n;
				}

				float s = 0, t = 0;
				for (int i = 0; i < nDims; i++){
					s += ins[i];
				}
				s *= F2;
				int inds[nDims];
				for (int i = 0; i < nDims; i++){
					inds[i] = fast_math::floorf(ins[i] + s);
					t += inds[i];
				}
				t *= G2;
				float d[nDims + 1][nDims];
				int sorted[nDims];
				//x0
				for (int i = 0; i < nDims; i++){
					d[0][i] = ins[i] - (inds[i] - t);
				}
				for (int i = 0; i < nDims; i++){
					int l = i;
					for (int j = i + 1; j < nDims; j++){
						if (d[0][l] < d[0][j]){ l = j; }
					}
					sorted[i] = l;
				}
				int ij1[nDims - 1][nDims] = { 0 };

				for (int i = 0; i < nDims - 1; i++){
					for (int j = i; j < nDims - 1; j++){
						ij1[j][sorted[i]] = 1;
					}
				}
				//x1-dims-2
				int gg = 1;
				for (int i = 0; i < nDims - 1; i++){
					for (int j = 0; j < nDims; j++){
						d[gg][j] = d[0][j] - ij1[i][j] + (gg*G2);
					}
					gg++;
				}
				//x1-dims-1
				for (int i = 0; i < nDims; i++){
					d[nDims][i] = d[0][i] - 1 + gg*G2;
				}
				int iijj[nDims];
				for (int i = 0; i < nDims; i++){
					iijj[i] = inds[i] & 255;
				}
				int gi[nDims + 1];
				float tyt[nDims + 1];
				int g = permi[iijj[nDims - 1]];
				for (int i = nDims - 2; i >= 0; i--){
					g = permi[iijj[i] + g];
				}
				gi[0] = ix(g % 12, 0, 3, 1);
				int ttg = 0;
				for (int i = 1; i < nDims; i++){
					g = permi[iijj[nDims - 1] + ij1[ttg][nDims - 1]];
					for (int j = nDims - 2; j >= 0; j--){
						g = permi[iijj[j] + ij1[ttg][j] + g];
					}
					ttg++;
					gi[i] = ix(g % 12, 0, 3, 1);
				}
				g = permi[iijj[nDims - 1] + 1];
				for (int j = nDims - 2; j >= 0; j--){
					g = permi[iijj[j] + 1 + g];
				}
				gi[nDims] = ix(g % 12, 0, 3, 1);

				//tyt[0] = 0.5f;
				//for (int j = 0; j < nDims; j++){
				//	tyt[0] -= d[0][j] * d[0][j];
				//}
				for (int i = 0; i < nDims + 1; i++){
					tyt[i] = 0.5f;
					for (int j = 0; j < nDims; j++){
						tyt[i] -= d[i][j] * d[i][j];
					}
				}
				float n[nDims + 1];
				for (int i = 0; i < nDims + 1; i++){
					if (tyt[i] < 0){ n[i] = 0.0f; }
					else{
						tyt[i] *= tyt[i];
						float dotf = 0;
						for (int j = 0; j < nDims; j++){
							dotf += gradsi[gi[i] + j] * d[i][j];
						}
						n[i] = tyt[i] * tyt[i] * dotf;
					}
				}
				float pval = 0;
				for (int i = 0; i < nDims+1; i++){
					pval += n[i];
				}
				pval = (70.0f*pval);
				val -= pval / o;
			}
		//	float x1 = i1 - 500, y1 = j1 - 500;
		//	float d = fast_math::sqrtf(x1*x1 + y1*y1);
			imgar[idx[0]][idx[1]] = fast_math::fabsf(val / nocts);
		});
	}
	//2 dimensional only
	void OLDcreatenoise(DATATYPE * data, int n, int nocts){
		array_view<DATATYPE, nDims> imgar(n, n, img);
		array_view<int, 1> permi(512, perm);
		array_view<float, 1> gradsi(36, grads3);
		float F2 = 0.5*(sqrt(3.0) - 1.0);
		float G2 = (3.0 - sqrt(3.0)) / 6.0;
		parallel_for_each(
			// Define the compute domain, which is the set of threads that are created.
			imgar.extent,
			// Define the code to run on each thread on the accelerator.
			[=](index<nDims> idx) restrict(amp)
		{
			float val = 1;
			////float ii1 = idx[0],// ixi(idx[0], 1000, 1),
			//	jj1 = idx[1];// ixj(idx[0], 1000, 1);
			for (int o = 1; o <= nocts; o++){
				float o2 = fast_math::pow(2, o - 1);
				float xin = idx[0] * o2;
				float yin = idx[1] * o2;
				xin /= 1000.0f;
				yin /= 1000.0f;
				float n0, n1, n2; // Noise contributions from the three corners
				////				 Skew the input space to determine which simplex cell we're in
				float s = (xin + yin)*F2; // Hairy factor for 2D
				int i = fast_math::floorf(xin + s);
				int j = fast_math::floorf(yin + s);
				float t = (i + j)*G2;
				float X0 = i - t; // Unskew the cell origin back to (x,y) space
				float Y0 = j - t;
				float x0 = xin - X0; // The x,y distances from the cell origin
				float y0 = yin - Y0;

				int i1, j1; // Offsets for second (middle) corner of simplex in (i,j) coords
				if (x0>y0) { i1 = 1; j1 = 0; } // lower triangle, XY order: (0,0)->(1,0)->(1,1)
				else { i1 = 0; j1 = 1; }      // upper triangle, YX order: (0,0)->(0,1)->(1,1)

				float x1 = x0 - i1 + G2; // Offsets for middle corner in (x,y) unskewed coords
				float y1 = y0 - j1 + G2;
				float x2 = x0 - 1.0f + 2.0f * G2; // Offsets for last corner in (x,y) unskewed coords
				float y2 = y0 - 1.0f + 2.0f * G2;
				////				 Work out the hashed gradient indices of the three simplex corners
				int ii = i & 255;
				int jj = j & 255;
				int gi0 = ix(permi[ii + permi[jj]] % 12, 0, 3, 1);
				int gi1 = ix(permi[ii + i1 + permi[jj + j1]] % 12, 0, 3, 1);
				int gi2 = ix(permi[ii + 1 + permi[jj + 1]] % 12, 0, 3, 1);
				////		 Calculate the contribution from the three corners
				float t0 = 0.5f - x0*x0 - y0*y0;
				if (t0<0) n0 = 0.0f;
				else {
					t0 *= t0;

					n0 = t0 * t0 * dot(gradsi[gi0], gradsi[gi0 + 1], x0, y0);  // (x,y) of grad3 used for 2D gradient
				}
				float t1 = 0.5f - x1*x1 - y1*y1;
				if (t1<0) n1 = 0.0f;
				else {
					t1 *= t1;
					n1 = t1 * t1 * dot(gradsi[gi1], gradsi[gi1 + 1], x1, y1);
				}
				float t2 = 0.5f - x2*x2 - y2*y2;
				if (t2<0) n2 = 0.0f;
				else {
					t2 *= t2;
					n2 = t2 * t2 * dot(gradsi[gi2], gradsi[gi2 + 1], x2, y2);
				}
				///		 Add contributions from each corner to get the final noise value.
				///		 The result is scaled to return values in the interval [-1,1].

				val -= (70.0f * (n0 + n1 + n2)) / o;
			}
			//	float x1 = i1 - 500, y1 = j1 - 500;
			//	float d = fast_math::sqrtf(x1*x1 + y1*y1);
			//if (val > 0){
			imgar[idx[0]][idx[1]] = fast_math::fabsf(val / nocts);
			//}
			//else{
			//	imgar[idx] = -(val / nocts);
			//}
		});
	}
				

	
};