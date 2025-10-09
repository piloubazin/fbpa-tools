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

	private int ndims = 10;
	private int nbins = 100;
		
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setEmbeddingImage(float[] val) { imgEmbedding = val; }
	
	
	public final void setDimensions(int val) { ndims = val; }
	public final void setCoordinateBins(int val) { nbins = val; }
					
	public final void setImageDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setImageDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setImageResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setImageResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
				
	// create outputs
	public final float[] 	getEmbeddedImage() { return embeddedImage; }
	

	public final void execute() {
	    
	    // create the output image
	    embeddedImage = new float[ndims*nbins];
	    float[] count = new float[ndims*nbins];
	    
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
	            if (imgEmbedding[xyz+d*nxyz]<cmax)
	                bin += Numerics.floor(nbins*(imgEmbedding[xyz+d*nxyz]-cmin)/(cmax-cmin)) + d*nbins;
	            else
	                bin += nbins-1 + d*nbins;
	        }
	        embeddedImage[bin] += inputImage[xyz];
	        count[bin]++;
	    }
	        
	    for (int b=0;b<nbins*ndims;b++) if (count[b]>0) {
	        embeddedImage[b] /= count[b];
	    }

	    return;
     }

}