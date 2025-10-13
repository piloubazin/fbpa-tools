package nl.fullbrainpicture.libraries;

import nl.fullbrainpicture.utilities.*;
import org.apache.commons.math3.util.FastMath;

/**
 *
 *  This class computes various basic image filters.
 *	
 *	@version    Feb 2011
 *	@author     Pierre-Louis Bazin
 *		
 *
 */

public class ImageFilters {
	
	// no data: used as a library of functions
	
    // simple image functions
    
    public static final byte X = 0;
    public static final byte Y = 1;
    public static final byte Z = 2;
    public static final byte T = 3;

	/**
	 *	convolution
	 */
	public static float[][][] convolution3D(float[][][] image, int nx, int ny, int nz, float[][][] kernel, int kx, int ky, int kz) {
		float[][][] result = new float[nx][ny][nz];
		int xi,yj,zl;
		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			result[x][y][z] = 0.0f;	
			for (int i=-kx;i<=kx;i++) for (int j=-ky;j<=ky;j++) for (int l=-kz;l<=kz;l++) {
				xi = x+i; yj = y+j; zl = z+l;
				if ( (xi>=0) && (xi<nx) && (yj>=0) && (yj<ny) && (zl>=0) && (zl<nz) ) {
					result[x][y][z] += image[xi][yj][zl]*kernel[kx+i][ky+j][kz+l];
				}
			}
		}

		return result;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz})
	 */
	public static float[][][] separableConvolution3D(float[][][] image, int nx, int ny, int nz, float[][] kernel, int kx, int ky, int kz) {
		float[][][] result = new float[nx][ny][nz];
		float[][][] temp = new float[nx][ny][nz];
		int xi,yj,zl;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			result[x][y][z] = 0.0f;	
			for (int i=-kx;i<=kx;i++) {
				if ( (x+i>=0) && (x+i<nx) ) {
					result[x][y][z] += image[x+i][y][z]*kernel[0][kx+i];
				}
			}
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			temp[x][y][z] = 0.0f;	
			for (int i=-ky;i<=ky;i++) {
				if ( (y+i>=0) && (y+i<ny) ) {
					temp[x][y][z] += result[x][y+i][z]*kernel[1][ky+i];
				}
			}
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			result[x][y][z] = 0.0f;	
			for (int i=-kz;i<=kz;i++) {
				if ( (z+i>=0) && (z+i<nz) ) {
					result[x][y][z] += temp[x][y][z+i]*kernel[2][kz+i];
				}
			}
		}
		temp = null;

		return result;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz})
	 */
	public static float[] separableConvolution3D(float[] image, int nx, int ny, int nz, float[][] kernel, int kx, int ky, int kz) {
		float[] result = new float[nx*ny*nz];
		float[] temp = new float[nx*ny*nz];
		int xi,yj,zl;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;	
			for (int i=-kx;i<=kx;i++) {
				if ( (x+i>=0) && (x+i<nx) ) {
					result[xyz] += image[xyz+i]*kernel[0][kx+i];
				}
			}
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			temp[xyz] = 0.0f;	
			for (int i=-ky;i<=ky;i++) {
				if ( (y+i>=0) && (y+i<ny) ) {
					temp[xyz] += result[xyz+i*nx]*kernel[1][ky+i];
				}
			}
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;	
			for (int i=-kz;i<=kz;i++) {
				if ( (z+i>=0) && (z+i<nz) ) {
					result[xyz] += temp[xyz+i*nx*ny]*kernel[2][kz+i];
				}
			}
		}
		temp = null;

		return result;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz})
	 */
	public static float[] separableConvolution3D(float[] image, int nx, int ny, int nz, float[][] kernel) {
		float[] result = new float[nx*ny*nz];
		float[] temp = new float[nx*ny*nz];
		int xi,yj,zl;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		
		int kx = (kernel[X].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;	
			for (int i=-kx;i<=kx;i++) {
				if ( (x+i>=0) && (x+i<nx) ) {
					result[xyz] += image[xyz+i]*kernel[0][kx+i];
				}
			}
		}
		int ky = (kernel[Y].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			temp[xyz] = 0.0f;	
			for (int i=-ky;i<=ky;i++) {
				if ( (y+i>=0) && (y+i<ny) ) {
					temp[xyz] += result[xyz+i*nx]*kernel[1][ky+i];
				}
			}
		}
		int kz = (kernel[Z].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;	
			for (int i=-kz;i<=kz;i++) {
				if ( (z+i>=0) && (z+i<nz) ) {
					result[xyz] += temp[xyz+i*nx*ny]*kernel[2][kz+i];
				}
			}
		}
		temp = null;

		return result;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky})
	 */
	public static float[] separableConvolution2D(float[] image, int nx, int ny, float[][] kernel) {
		float[] result = new float[nx*ny];
		float[] temp = new float[nx*ny];
		int xi,yj;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		
		int kx = (kernel[X].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
			int xy = x+nx*y;
			result[xy] = 0.0f;	
			for (int i=-kx;i<=kx;i++) {
				if ( (x+i>=0) && (x+i<nx) ) {
					result[xy] += image[xy+i]*kernel[0][kx+i];
				}
			}
		}
		int ky = (kernel[Y].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
			int xy = x+nx*y;
			temp[xy] = 0.0f;	
			for (int i=-ky;i<=ky;i++) {
				if ( (y+i>=0) && (y+i<ny) ) {
					temp[xy] += result[xy+i*nx]*kernel[1][ky+i];
				}
			}
		}

		return temp;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz,kt})
	 */
	public static float[] separableConvolution4D(float[] image, int nx, int ny, int nz, int nt, float[][] kernel) {
		float[] result = new float[nx*ny*nz*nt];
		float[] temp = new float[nx*ny*nz*nt];
		
		int kx = (kernel[X].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) for (int t=0;z<nt;t++) {
			int xyzt = x+nx*y+nx*ny*z+nx*ny*nz*t;
			result[xyzt] = 0.0f;	
			for (int i=-kx;i<=kx;i++) {
				if ( (x+i>=0) && (x+i<nx) ) {
					result[xyzt] += image[xyzt+i]*kernel[X][kx+i];
				}
			}
		}
		int ky = (kernel[Y].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) for (int t=0;z<nt;t++) {
			int xyzt = x+nx*y+nx*ny*z+nx*ny*nz*t;
			temp[xyzt] = 0.0f;	
			for (int i=-ky;i<=ky;i++) {
				if ( (y+i>=0) && (y+i<ny) ) {
					temp[xyzt] += result[xyzt+i*nx]*kernel[Y][ky+i];
				}
			}
		}
		int kz = (kernel[Z].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) for (int t=0;z<nt;t++) {
			int xyzt = x+nx*y+nx*ny*z+nx*ny*nz*t;
			result[xyzt] = 0.0f;	
			for (int i=-kz;i<=kz;i++) {
				if ( (z+i>=0) && (z+i<nz) ) {
					result[xyzt] += temp[xyzt+i*nx*ny]*kernel[Z][kz+i];
				}
			}
		}
		int kt = (kernel[T].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) for (int t=0;z<nt;t++) {
			int xyzt = x+nx*y+nx*ny*z+nx*ny*nz*t;
			temp[xyzt] = 0.0f;	
			for (int i=-kt;i<=kt;i++) {
				if ( (t+i>=0) && (t+i<nt) ) {
					temp[xyzt] += result[xyzt+i*nx*ny*nz]*kernel[T][kt+i];
				}
			}
		}
		result = null;

		return temp;
	}
		
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz})
	*	this method compensates for boundaries 
	 */
	public static float[] separableBoundedConvolution3D(float[] image, int nx, int ny, int nz, float[][] kernel) {
		float[] result = new float[nx*ny*nz];
		float[] temp = new float[nx*ny*nz];
		int xi,yj,zl;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		int kx = (kernel[X].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;
			float den = 0.0f;
			for (int i=-kx;i<=kx;i++) {
				if ( (x+i>=0) && (x+i<nx) ) {
					result[xyz] += image[xyz+i]*kernel[X][kx+i];
					den += kernel[X][kx+i];
				}
			}
			result[xyz] /= den;
		}
		int ky = (kernel[Y].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			temp[xyz] = 0.0f;	
			float den = 0.0f;
			for (int i=-ky;i<=ky;i++) {
				if ( (y+i>=0) && (y+i<ny) ) {
					temp[xyz] += result[xyz+i*nx]*kernel[Y][ky+i];
					den += kernel[Y][ky+i];
				}
			}
			result[xyz] /= den;
		}
		int kz = (kernel[Z].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;	
			float den = 0.0f;
			for (int i=-kz;i<=kz;i++) {
				if ( (z+i>=0) && (z+i<nz) ) {
					result[xyz] += temp[xyz+i*nx*ny]*kernel[Z][kz+i];
					den += kernel[Z][kz+i];
				}
			}
			result[xyz] /= den;
		}
		temp = null;

		return result;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz})
	*	this method compensates for masked regions 
	 */
	public static float[] separableMaskedConvolution3D(float[] image, boolean[] mask, int nx, int ny, int nz, float[][] kernel) {
		float[] result = new float[nx*ny*nz];
		float[] temp = new float[nx*ny*nz];
		int xi,yj,zl;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		int kx = (kernel[X].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;
			double num = 0.0;
			double den = 0.0;
			
			if (mask[xyz]){
				for (int i=-kx;i<=kx;i++) {
					if ( (x+i>=0) && (x+i<nx) && mask[xyz+i] ) {
						num += image[xyz+i]*kernel[X][kx+i];
						den += kernel[X][kx+i];
					}
				}
				if (den*den>0) num /= den;
				result[xyz] = (float)num;
			}
		}
		
		int ky = (kernel[Y].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			temp[xyz] = 0.0f;
			double num = 0.0;
			double den = 0.0;
			
			if (mask[xyz]){
				for (int i=-ky;i<=ky;i++) {
					if ( (y+i>=0) && (y+i<ny) && mask[xyz+nx*i] ) {
						num += result[xyz+i*nx]*kernel[Y][ky+i];
						den += kernel[Y][ky+i];
					}
				}
				if (den*den>0) num /= den;
				temp[xyz] = (float)num;
			}
		}
		
		int kz = (kernel[Z].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;	
			double num = 0.0;
			double den = 0.0;
			
			if (mask[xyz]){
				for (int i=-kz;i<=kz;i++) {
					if ( (z+i>=0) && (z+i<nz) && mask[xyz+nx*ny*i] ) {
						num += temp[xyz+i*nx*ny]*kernel[Z][kz+i];
						den += kernel[Z][kz+i];
					}
				}
				if (den*den>0) num /= den;
				result[xyz] = (float)num;
			}
		}
		temp = null;

		return result;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz})
	*	this method compensates for masked regions 
	 */
	public static float[][][] separableMaskedConvolution3D(float[][][] image, boolean[][][] mask, int nx, int ny, int nz, float[][] kernel) {
		float[][][] result = new float[nx][ny][nz];
		float[][][] temp = new float[nx][ny][nz];
		int xi,yj,zl;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		int kx = (kernel[X].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			result[x][y][z] = 0.0f;
			double num = 0.0;
			double den = 0.0;
			
			if (mask[x][y][z]){
				for (int i=-kx;i<=kx;i++) {
					if ( (x+i>=0) && (x+i<nx) && mask[x+i][y][z] ) {
						num += image[x+i][y][z]*kernel[X][kx+i];
						den += kernel[X][kx+i];
					}
				}
				if (den*den>0) num /= den;
				result[x][y][z] = (float)num;
			}
		}
		
		int ky = (kernel[Y].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			temp[x][y][z] = 0.0f;
			double num = 0.0;
			double den = 0.0;
			
			if (mask[x][y][z]){
				for (int i=-ky;i<=ky;i++) {
					if ( (y+i>=0) && (y+i<ny) && mask[x][y+i][z] ) {
						num += result[x][y+i][z]*kernel[Y][ky+i];
						den += kernel[Y][ky+i];
					}
				}
				if (den*den>0) num /= den;
				temp[x][y][z] = (float)num;
			}
		}
		
		int kz = (kernel[Z].length-1)/2;
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			result[x][y][z] = 0.0f;	
			double num = 0.0;
			double den = 0.0;
			
			if (mask[x][y][z]){
				for (int i=-kz;i<=kz;i++) {
					if ( (z+i>=0) && (z+i<nz) && mask[x][y][z+i] ) {
						num += temp[x][y][z+i]*kernel[Z][kz+i];
						den += kernel[Z][kz+i];
					}
				}
				if (den*den>0) num /= den;
				result[x][y][z] = (float)num;
			}
		}
		temp = null;

		return result;
	}
		
	/**
	*	convolution with a separable kernel (the kernel is 3x{kx,ky,kz})
	*	this method compensates for masked regions 
	 */
	public static float[] separableMaskedConvolution3D(float[] image, boolean[] mask, int nx, int ny, int nz, float[][] kernel, int kx, int ky, int kz) {
		float[] result = new float[nx*ny*nz];
		float[] temp = new float[nx*ny*nz];
		int xi,yj,zl;
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			double num = 0.0;
			double den = 0.0;
			for (int i=-kx;i<=kx;i++) {
				if ( (x+i>=0) && (x+i<nx) && mask[xyz+i] ) {
					num += image[xyz+i]*kernel[X][kx+i];
					den += kernel[X][kx+i];
				}
			}
			if (den*den>0) num /= den;
			result[xyz] = (float)num;
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			double num = 0.0;
			double den = 0.0;
			for (int i=-ky;i<=ky;i++) {
				if ( (y+i>=0) && (y+i<ny) && mask[xyz+nx*i] ) {
					num += result[xyz+i*nx]*kernel[Y][ky+i];
					den += kernel[Y][ky+i];
				}
			}
			if (den*den>0) num /= den;
			temp[xyz] = (float)num;
		}
		for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
			int xyz = x+nx*y+nx*ny*z;
			result[xyz] = 0.0f;	
			double num = 0.0;
			double den = 0.0;
			for (int i=-kz;i<=kz;i++) {
				if ( (z+i>=0) && (z+i<nz) && mask[xyz+nx*ny*i] ) {
					num += temp[xyz+i*nx*ny]*kernel[Z][kz+i];
					den += kernel[Z][kz+i];
				}
			}
			if (den*den>0) num /= den;
			result[xyz] = (float)num;
		}
		temp = null;

		return result;
	}
		
	/**
	 *	Gaussian kernel
	 */
	public static float[][][] gaussianKernel3D(float sx, float sy, float sz) {
		int kx,ky,kz;
		float sum;
		
		// kernel size
		kx = Numerics.ceil(Numerics.max(3.0f*sx-0.5f,0.0f));
		ky = Numerics.ceil(Numerics.max(3.0f*sy-0.5f,0.0f));
		kz = Numerics.ceil(Numerics.max(3.0f*sz-0.5f,0.0f));
		
		// create the kernel
		float[][][] kernel = new float[2*kx+1][2*ky+1][2*kz+1];
		sum = 0.0f;
		for (int i=-kx;i<=kx;i++) for (int j=-ky;j<=ky;j++) for (int l=-kz;l<=kz;l++) {
			kernel[kx+i][ky+j][kz+l] = (float)Math.exp( - 0.5f*(i*i)/(sx*sx) + (j*j)/(sy*sy) + (l*l)/(sz*sz) );
			sum += kernel[kx+i][ky+j][kz+l];
		}
		// normalize
		for (int i=-kx;i<=kx;i++) for (int j=-ky;j<=ky;j++) for (int l=-kz;l<=kz;l++) {
			kernel[kx+i][ky+j][kz+l] = kernel[kx+i][ky+j][kz+l]/sum;
		}

		return kernel;
	}
	/**
	 *	Gaussian kernel for separable convolution
	 */
	public static float[][] separableGaussianKernel3D(float sx, float sy, float sz) {
		int kx,ky,kz;
		float sum;
		
		// kernel size
		kx = Numerics.ceil(Math.max(3.0f*sx-0.5f,0.0f));
		ky = Numerics.ceil(Math.max(3.0f*sy-0.5f,0.0f));
		kz = Numerics.ceil(Math.max(3.0f*sz-0.5f,0.0f));
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		//MedicUtilPublic.displayMessage("scale: "+sx+"x"+sy+"x"+sz+"\n");
		// create the kernel
		float[][] kernel = new float[3][];
		kernel[0] = new float[2*kx+1]; 
		kernel[1] = new float[2*ky+1]; 
		kernel[2] = new float[2*kz+1]; 
		
		sum = 0.0f;
		for (int i=-kx;i<=kx;i++) {
			kernel[0][kx+i] = (float)Math.exp( - 0.5f*(i*i)/(sx*sx) );
			//MedicUtilPublic.displayMessage("exp("+( - 0.5f*(i*i)/(sx*sx) )+") = "+kernel[0][kx+i]+"\n");
			sum += kernel[0][kx+i];
		}
		for (int i=-kx;i<=kx;i++) kernel[0][kx+i] /= sum;
		
		sum = 0.0f;
		for (int j=-ky;j<=ky;j++) {
			kernel[1][ky+j] = (float)Math.exp( - 0.5f*(j*j)/(sy*sy) );
			sum += kernel[1][ky+j];
		}
		for (int j=-ky;j<=ky;j++) kernel[1][ky+j] /= sum;
		
		sum = 0.0f;
		for (int l=-kz;l<=kz;l++) {
			kernel[2][kz+l] = (float)Math.exp( - 0.5f*(l*l)/(sz*sz) );
			sum += kernel[2][kz+l];
		}
		for (int l=-kz;l<=kz;l++) kernel[2][kz+l] /= sum;
		
		return kernel;
	}
	
		/**
	 *	Gaussian kernel for separable convolution
	 */
	public static float[][] separableGaussianKernel2D(float sx, float sy) {
		int kx,ky;
		float sum;
		
		// kernel size
		kx = Numerics.ceil(Math.max(3.0f*sx-0.5f,0.0f));
		ky = Numerics.ceil(Math.max(3.0f*sy-0.5f,0.0f));
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		//MedicUtilPublic.displayMessage("scale: "+sx+"x"+sy+"x"+sz+"\n");
		// create the kernel
		float[][] kernel = new float[2][];
		kernel[0] = new float[2*kx+1]; 
		kernel[1] = new float[2*ky+1]; 
		
		sum = 0.0f;
		for (int i=-kx;i<=kx;i++) {
			kernel[0][kx+i] = (float)Math.exp( - 0.5f*(i*i)/(sx*sx) );
			//MedicUtilPublic.displayMessage("exp("+( - 0.5f*(i*i)/(sx*sx) )+") = "+kernel[0][kx+i]+"\n");
			sum += kernel[0][kx+i];
		}
		for (int i=-kx;i<=kx;i++) kernel[0][kx+i] /= sum;
		
		sum = 0.0f;
		for (int j=-ky;j<=ky;j++) {
			kernel[1][ky+j] = (float)Math.exp( - 0.5f*(j*j)/(sy*sy) );
			sum += kernel[1][ky+j];
		}
		for (int j=-ky;j<=ky;j++) kernel[1][ky+j] /= sum;
		
		return kernel;
	}
	/**
	 *	Gaussian kernel for separable convolution
	 */
	public static float[][] separableGaussianKernel4D(float sx, float sy, float sz, float st) {
		int kx,ky,kz,kt;
		float sum;
		
		// kernel size
		kx = Numerics.ceil(Math.max(3.0f*sx-0.5f,0.0f));
		ky = Numerics.ceil(Math.max(3.0f*sy-0.5f,0.0f));
		kz = Numerics.ceil(Math.max(3.0f*sz-0.5f,0.0f));
		kt = Numerics.ceil(Math.max(3.0f*st-0.5f,0.0f));
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		//MedicUtilPublic.displayMessage("scale: "+sx+"x"+sy+"x"+sz+"\n");
		// create the kernel
		float[][] kernel = new float[4][];
		kernel[0] = new float[2*kx+1]; 
		kernel[1] = new float[2*ky+1]; 
		kernel[2] = new float[2*kz+1]; 
		kernel[3] = new float[2*kt+1]; 
		
		sum = 0.0f;
		for (int i=-kx;i<=kx;i++) {
			kernel[0][kx+i] = (float)Math.exp( - 0.5f*(i*i)/(sx*sx) );
			//MedicUtilPublic.displayMessage("exp("+( - 0.5f*(i*i)/(sx*sx) )+") = "+kernel[0][kx+i]+"\n");
			sum += kernel[0][kx+i];
		}
		for (int i=-kx;i<=kx;i++) kernel[0][kx+i] /= sum;
		
		sum = 0.0f;
		for (int j=-ky;j<=ky;j++) {
			kernel[1][ky+j] = (float)Math.exp( - 0.5f*(j*j)/(sy*sy) );
			sum += kernel[1][ky+j];
		}
		for (int j=-ky;j<=ky;j++) kernel[1][ky+j] /= sum;
		
		sum = 0.0f;
		for (int l=-kz;l<=kz;l++) {
			kernel[2][kz+l] = (float)Math.exp( - 0.5f*(l*l)/(sz*sz) );
			sum += kernel[2][kz+l];
		}
		for (int l=-kz;l<=kz;l++) kernel[2][kz+l] /= sum;
		
		sum = 0.0f;
		for (int m=-kt;m<=kt;m++) {
			kernel[3][kt+m] = (float)Math.exp( - 0.5f*(m*m)/(st*st) );
			sum += kernel[3][kt+m];
		}
		for (int m=-kt;m<=kt;m++) kernel[3][kt+m] /= sum;
		
		return kernel;
	}
	
	public static float[][] separableRandomKernel3D(float sx, float sy, float sz) {
		int kx,ky,kz;
		float sum;
		
		// kernel size
		kx = Numerics.ceil(Math.max(3.0f*sx-0.5f,0.0f));
		ky = Numerics.ceil(Math.max(3.0f*sy-0.5f,0.0f));
		kz = Numerics.ceil(Math.max(3.0f*sz-0.5f,0.0f));
		
		//MedicUtilPublic.displayMessage("kernel size: "+kx+"x"+ky+"x"+kz+"\n");
		//MedicUtilPublic.displayMessage("scale: "+sx+"x"+sy+"x"+sz+"\n");
		// create the kernel
		float[][] kernel = new float[3][];
		kernel[0] = new float[2*kx+1]; 
		kernel[1] = new float[2*ky+1]; 
		kernel[2] = new float[2*kz+1]; 
		
		sum = 0.0f;
		for (int i=-kx;i<=kx;i++) {
			kernel[0][kx+i] = (float)FastMath.random();
			sum += kernel[0][kx+i];
		}
		for (int i=-kx;i<=kx;i++) kernel[0][kx+i] /= sum;
		
		sum = 0.0f;
		for (int j=-ky;j<=ky;j++) {
			kernel[1][ky+j] = (float)FastMath.random();
			sum += kernel[1][ky+j];
		}
		for (int j=-ky;j<=ky;j++) kernel[1][ky+j] /= sum;
		
		sum = 0.0f;
		for (int l=-kz;l<=kz;l++) {
			kernel[2][kz+l] = (float)FastMath.random();
			sum += kernel[2][kz+l];
		}
		for (int l=-kz;l<=kz;l++) kernel[2][kz+l] /= sum;
		
		return kernel;
	}
	
}
