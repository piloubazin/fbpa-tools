package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.HashSet;

/*
 * @author Pierre-Louis Bazin
 */
public class OctContrastIntegration {

	// jist containers
	private float[] inputImage=null;
	private float[] weightImage=null;
	private int[] maskImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	
	private String contrastType;
	private String weightingType;
	private float ratio = 0.25f;
	
	private float[] contrastImage;
		
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);

	// direction labeling		
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
	public	static	final	byte	Z = 2;
	public	static	final	byte	T = 3;

	// computation variables
	
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setInputImage(float[] val) { inputImage = val; }
	public final void setWeightImage(float[] val) { weightImage = val; }
	public final void setMaskImage(int[] val) { maskImage = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
	
	public final void setContrastType(String val) { contrastType = val; }
	public final void setWeightingType(String val) { weightingType = val; }
	public final void setRatio(float val) { ratio = val; }
	
	// create outputs
	public final float[] getContrastImage() { return contrastImage; }
	
	public void execute(){
	    
	    // make mask
	    boolean[] mask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mask[xyz] = true;
		    if (inputImage[xyz]==0) mask[xyz] = false;
		    if (maskImage!=null && maskImage[xyz]==0) mask[xyz] = false;
		}
		maskImage = null;
	    
		// check on desired contrast
		if (contrastType.equals("birefringence")) {
            System.out.println("Compute birefringence model");
            birefringenceEstimation();
        } else if (contrastType.equals("orientation")) {
            System.out.println("Compute orientation model");
            orientationSumEstimation();
        }
	}
	
	private void birefringenceEstimation() {

	    // estimate start / stop location on weights
	    float wmax = 0.0f;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        if (weightImage[xyz]>wmax) wmax = weightImage[xyz];
	    }
	    boolean[] obj = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        obj[xyz] = (weightImage[xyz]>ratio*wmax);
	    }
	    obj = ObjectLabeling.largestObject(obj, nx, ny, nz, 26);
	    
	    // rescale weights as desired
	    if (weightingType.equals("constant")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = 1.0f;
	            else weightImage[xyz] = 0.0f;
	        }
	    } else if (weightingType.equals("log")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = (float)FastMath.log(1.0+weightImage[xyz]);
	            else weightImage[xyz] = 0.0f;
	        }
	    } else if (weightingType.equals("exp")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = (float)FastMath.exp(weightImage[xyz])-1.0f;
	            else weightImage[xyz] = 0.0f;
	        }
	    }
	    
	    // use selected weights only
	    float[] slope = new float[nx*ny];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	        double meandepth = 0.0;
	        double meanvalue = 0.0;
	        double meanweight = 0.0;
	        for (int z=0;z<nz;z++) {
	            int xyz = x+nx*y+nx*ny*z;
	            if (obj[xyz]) {
	                meandepth += weightImage[xyz]*z;
	                meanvalue += weightImage[xyz]*inputImage[xyz];
	                meanweight += weightImage[xyz];
	            }
	        }
	        if (meanweight>0) {
                meandepth /= meanweight;
                meanvalue /= meanweight;
	        
                double num = 0.0;
                double den = 0.0;
                for (int z=0;z<nz;z++) {
                    int xyz = x+nx*y+nx*ny*z;
                    if (obj[xyz]) {
                        num += weightImage[xyz]*(inputImage[xyz]-meanvalue)
                              *weightImage[xyz]*(z-meandepth);
                        den += weightImage[xyz]*(z-meandepth)
                              *weightImage[xyz]*(z-meandepth);
                    }
                }
                if (den>0) {
                    slope[x+nx*y] = (float)(num/den);
                }
            }
        }
        contrastImage = slope;
        
	    return;
	}
	
	private void orientationEstimation() {
	    int nbin = 36;
	    
	    // estimate start / stop location on weights
	    float wmax = 0.0f;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        if (weightImage[xyz]>wmax) wmax = weightImage[xyz];
	    }
	    boolean[] obj = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        obj[xyz] = (weightImage[xyz]>ratio*wmax);
	    }
	    obj = ObjectLabeling.largestObject(obj, nx, ny, nz, 26);
	    
	    // rescale weights as desired
	    if (weightingType.equals("constant")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = 1.0f;
	            else weightImage[xyz] = 0.0f;
	        }
	    } else if (weightingType.equals("log")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = (float)FastMath.log(1.0+weightImage[xyz]);
	            else weightImage[xyz] = 0.0f;
	        }
	    } else if (weightingType.equals("exp")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = (float)FastMath.exp(weightImage[xyz])-1.0f;
	            else weightImage[xyz] = 0.0f;
	        }
	    }
	    
	    // use selected weights only
	    float[] orient = new float[nx*ny];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	        // angle histogram binned at 5 deg, values in -90,90 deg
	        double[] hist = new double[nbin];
	        
	        for (int z=0;z<nz;z++) {
	            int xyz = x+nx*y+nx*ny*z;
	            if (obj[xyz]) {
	                int bin = Numerics.floor(inputImage[xyz]/FastMath.PI*nbin);
	                hist[bin] += weightImage[xyz];
	            }
	        }
	        int best=0;
	        for (int b=0;b<nbin;b++) {
	            if (hist[b]>hist[best]) best = b;
	        }
	        orient[x+nx*y] = (float)((best+0.5)*FastMath.PI/nbin);
	        
        }
        contrastImage = orient;
	    
        return;
	}

	
	private void orientationSumEstimation() {
	    
	    // estimate start / stop location on weights
	    float wmax = 0.0f;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        if (weightImage[xyz]>wmax) wmax = weightImage[xyz];
	    }
	    boolean[] obj = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        obj[xyz] = (weightImage[xyz]>ratio*wmax);
	    }
	    obj = ObjectLabeling.largestObject(obj, nx, ny, nz, 26);
	    
	    // rescale weights as desired
	    if (weightingType.equals("constant")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = 1.0f;
	            else weightImage[xyz] = 0.0f;
	        }
	    } else if (weightingType.equals("log")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = (float)FastMath.log(1.0+weightImage[xyz]);
	            else weightImage[xyz] = 0.0f;
	        }
	    } else if (weightingType.equals("exp")) {
	        for (int xyz=0;xyz<nxyz;xyz++) {
	            if (obj[xyz]) weightImage[xyz] = (float)FastMath.exp(weightImage[xyz])-1.0f;
	            else weightImage[xyz] = 0.0f;
	        }
	    }
	    
	    // use selected weights only
	    float[] orient = new float[nx*ny];
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	        // instead of histogram binning, take the weighted average of values
	        
	        // need to check at each step which of theta, theta-pi, theta+pi minimizes the error
	        double sum=0.0;
	        double sq=0.0;
	        double wgt=0.0;
	        for (int z=0;z<nz;z++) {
	            int xyz = x+nx*y+nx*ny*z;
	            if (obj[xyz]) {
	                if (wgt==0) {
	                    sum += weightImage[xyz]*inputImage[xyz];
	                    sq += weightImage[xyz]*inputImage[xyz]*inputImage[xyz];
	                    wgt += weightImage[xyz];
	                } else {
	                    double var0,varP,varM;
	                    var0 = (sq+weightImage[xyz]*inputImage[xyz]*inputImage[xyz])/(wgt+weightImage[xyz])
	                           -Numerics.square( (sum+weightImage[xyz]*inputImage[xyz])/(wgt+weightImage[xyz]) );
	                           
	                    varP = (sq+weightImage[xyz]*(inputImage[xyz]+FastMath.PI)*(inputImage[xyz]+FastMath.PI))/(wgt+weightImage[xyz])
	                           -Numerics.square( (sum+weightImage[xyz]*(inputImage[xyz]+FastMath.PI))/(wgt+weightImage[xyz]) );
	                           
	                    varM = (sq+weightImage[xyz]*(inputImage[xyz]-FastMath.PI)*(inputImage[xyz]-FastMath.PI))/(wgt+weightImage[xyz])
	                           -Numerics.square( (sum+weightImage[xyz]*(inputImage[xyz]-FastMath.PI))/(wgt+weightImage[xyz]) );
	                           
	                    if (var0<=varP && var0<=varM) {        
                            sum += weightImage[xyz]*inputImage[xyz];
                            sq += weightImage[xyz]*inputImage[xyz]*inputImage[xyz];
                            wgt += weightImage[xyz];
                        } else if (varP<=varM && varP<=var0) {
                            sum += weightImage[xyz]*(inputImage[xyz]+FastMath.PI);
                            sq += weightImage[xyz]*(inputImage[xyz]+FastMath.PI)*(inputImage[xyz]+FastMath.PI);
                            wgt += weightImage[xyz];
                        } else {
                            sum += weightImage[xyz]*(inputImage[xyz]-FastMath.PI);
                            sq += weightImage[xyz]*(inputImage[xyz]-FastMath.PI)*(inputImage[xyz]-FastMath.PI);
                            wgt += weightImage[xyz];
                        }
	                }	                    
	            }
	        }
	        double theta = sum/wgt;
	        if (theta>FastMath.PI) theta -= FastMath.PI;
	        if (theta<0) theta += FastMath.PI;
	        
	        orient[x+nx*y] = (float)theta;
	        
        }
        contrastImage = orient;
	    
        return;
	}

}
