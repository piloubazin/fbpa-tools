package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.linear.*;
//import Jama.*;
//import org.jblas.*;

import java.util.*;

/*
 * @author Pierre-Louis Bazin
 */
public class SpectralVoxelMapping {

    
	// jist containers
    private float[] inputImage;
    private float[] imgEmbedding;
    
    private float[] embeddedImage;
    
    private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private int ex, ey, ez, et;
	
	private int ndims = 3;
	private int[] nbins = null;
	private float[] smooth = null;
		
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setEmbeddingImage(float[] val) { imgEmbedding = val; }
	
	
	public final void setDimensions(int val) { 
	    ndims = val; 
	    nbins = new int[ndims];
	    smooth = new float[ndims];
	}
	public final void setBinAt(int n, int val) { nbins[n] = val; }
	public final void setSmoothAt(int n, float val) { smooth[n] = val; }
					
	public final void setImageDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setImageDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setImageResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setImageResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] 	getEmbeddedImage() { return embeddedImage; }
	
	public final int[] getEmbeddingDims() { 
	    if (ndims==1) return new int[] {ex}; 
	    else if (ndims==2) return new int[] {ex,ey}; 
	    else if (ndims==3) return new int[] {ex,ey,ez}; 
	    else return new int[] {ex,ey,ez,et}; 
	} 

	public final void execute() {
	    
	    // embedding dimensions, adding smoothing borders if needed
	    boolean doSmooth=false;
	    for (int d=0;d<ndims;d++) if (smooth[d]>0) doSmooth=true;
	    
	    int ox=0, oy=0, oz=0, ot=0;
	    if (doSmooth) {
	        ox = Numerics.ceil(2.0f*smooth[0]);
	        if (ndims>1) oy = Numerics.ceil(2.0f*smooth[1]);
	        if (ndims>2) oz = Numerics.ceil(2.0f*smooth[2]);
            if (ndims>3) ot = Numerics.ceil(2.0f*smooth[3]);
        }
	    ex = nbins[0]+2*ox;
	    ey = 0;
	    if (ndims>1) ey=nbins[1]+2*oy;
	    ez = 0;
	    if (ndims>2) ez=nbins[2]+2*oz;
	    et = 0;
	    if (ndims>3) et=nbins[3]+2*ot;

	    int ntotal = ex;
	    if (ndims>1) ntotal *= ey;
	    if (ndims>2) ntotal *= ez;
	    if (ndims>3) ntotal *= et;
	    
	    // create the output image
	    embeddedImage = new float[ntotal];
	    float[] count = new float[ntotal];
	    
	    // compute coordinate bounds (centered on zero)
	    float cmin = 1e9f;
	    float cmax = -1e9f;
	    for (int d=0;d<ndims;d++) for (int xyz=0;xyz<nxyz;xyz++) {
	        if (imgEmbedding[xyz+d*nxyz]<cmin) cmin = imgEmbedding[xyz+d*nxyz];
	        if (imgEmbedding[xyz+d*nxyz]>cmax) cmax = imgEmbedding[xyz+d*nxyz];
	    }
	    System.out.println("coordinates range: ["+cmin+", "+cmax+"]");
	    	    
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        int bin=0;
	        for (int d=0;d<ndims;d++) {
	            int n=nbins[d]-1;
	            if (imgEmbedding[xyz+d*nxyz]<cmax)
	                 n = Numerics.floor(nbins[d]*(imgEmbedding[xyz+d*nxyz]-cmin)/(cmax-cmin));
                if (d==0) bin += (n+ox);
                if (d==1) bin += (n+oy)*ex;
                if (d==2) bin += (n+oz)*ex*ey;
                if (d==3) bin += (n+ot)*ex*ey*ez;
	        }
	        embeddedImage[bin] += inputImage[xyz];
	        count[bin]++;
	    }
	        
	    for (int b=0;b<ntotal;b++) if (count[b]>0) {
	        embeddedImage[b] /= count[b];
	    }
	    
        if (doSmooth) {
            // smoothing, if needed
            if (ndims==2) {
                float[][] kernel = ImageFilters.separableGaussianKernel2D(smooth[0],smooth[1]);
                embeddedImage = ImageFilters.separableConvolution2D(embeddedImage, ex, ey, kernel);
            } else if (ndims==3) {
                float[][] kernel = ImageFilters.separableGaussianKernel3D(smooth[0],smooth[1],smooth[2]);
                embeddedImage = ImageFilters.separableConvolution3D(embeddedImage, ex, ey, ez, kernel);
            } else if (ndims==4) {
                float[][] kernel = ImageFilters.separableGaussianKernel4D(smooth[0],smooth[1],smooth[2],smooth[3]);
                embeddedImage = ImageFilters.separableConvolution4D(embeddedImage, ex, ey, ez, et, kernel);
            }
                
	    }

	    return;
     }

}