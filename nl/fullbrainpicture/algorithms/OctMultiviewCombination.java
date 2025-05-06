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
public class OctMultiviewCombination {

	// jist containers
	private float[][] inputImages=null;
	private float[][] inputOrient=null;
	private int[] maskImage=null;
	
	private int nc;
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	
	private float[] angle=null;
	
	private float[] directionImage;
		
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
	public final void setImageNumber(int val) {
	    nc = val;
	    inputImages = new float[nc][];
	    inputOrient = new float[nc][];
	    angle = new float[nc];
	}
	public final void setInputImageAt(int num, float[] val) { inputImages[num] = val; }
	public final void setOrientImageAt(int num, float[] val) { inputOrient[num] = val; }
	public final void setMaskImage(int[] val) { maskImage = val; }

	public final void setAngleAt(int num, float val) { angle[num] = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
	
	// create outputs
	public final float[] getDirectionImage() { return directionImage; }
	
	public void execute(){
	    
	    // make mask
	    boolean[] mask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mask[xyz] = true;
		    for (int c=0;c<nc;c++) {
                if (inputImages[c][xyz]==0) mask[xyz] = false;
            }
		    if (maskImage!=null && maskImage[xyz]==0) mask[xyz] = false;
		}
		maskImage = null;
	    
		directionImage = new float[nxyz*3];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    
		    int navg=0;
		    for (int c1=0;c1<nc;c1++) for (int c2=c1+1;c2<nc;c2++) {
		        // define the direction vectors
		        double[] v1 = new double[3];
		        double[] v2 = new double[3];
		        double[] n0 = new double[3];
		    
		        v1[X] = FastMath.cos(angle[c1])*FastMath.sin(inputOrient[c1][xyz]);
		        v1[Y] = -FastMath.cos(inputOrient[c1][xyz]);
		        v1[Z] = FastMath.sin(angle[c1])*FastMath.sin(inputOrient[c1][xyz]);
		    
		        v2[X] = FastMath.cos(angle[c2])*FastMath.sin(inputOrient[c2][xyz]);
		        v2[Y] = -FastMath.cos(inputOrient[c2][xyz]);
		        v2[Z] = FastMath.sin(angle[c2])*FastMath.sin(inputOrient[c2][xyz]);
		        
                n0[X] = v1[Y]*v2[Z] - v1[Z]*v2[Y];
                n0[Y] = v1[Z]*v2[X] - v1[X]*v2[Z];
                n0[Z] = v1[X]*v2[Y] - v1[Y]*v2[X];
		        
                // find the norm
                double n1 = FastMath.sqrt(Numerics.square(n0[X]*FastMath.cos(angle[c1])-n0[Z]*FastMath.sin(angle[c1]))
                                         +Numerics.square(n0[Y]));
                           
                double n2 = FastMath.sqrt(Numerics.square(n0[X]*FastMath.cos(angle[c2])-n0[Z]*FastMath.sin(angle[c2]))
                                         +Numerics.square(n0[Y]));
                           
                double alpha = (n1*inputImages[c1][xyz]+n2*inputImages[c2][xyz])/(n1*n1+n2*n2);           
	    
                // largest coordinate is always positive (arbitrary)
                       if (n0[X]<0 && n0[X]*n0[X]>=n0[Y]*n0[Y] && n0[X]*n0[X]>=n0[Z]*n0[Z]) {
                    n0[X] = -n0[X];
                    n0[Y] = -n0[Y];
                    n0[Z] = -n0[Z];
                } else if (n0[Y]<0 && n0[Y]*n0[Y]>=n0[Z]*n0[Z] && n0[Y]*n0[Y]>=n0[X]*n0[X]) {
                    n0[Y] = -n0[Y];
                    n0[Z] = -n0[Z];
                    n0[X] = -n0[X]; 
                } else if (n0[Z]<0 && n0[Z]*n0[Z]>=n0[X]*n0[X] && n0[Z]*n0[Z]>=n0[Y]*n0[Y]) {
                    n0[Z] = -n0[Z];
                    n0[X] = -n0[X];
                    n0[Y] = -n0[Y];
                }   
                
                directionImage[xyz+X*nxyz] += alpha*n0[X];
                directionImage[xyz+Y*nxyz] += alpha*n0[Y];
                directionImage[xyz+Z*nxyz] += alpha*n0[Z];
                navg++;
            }
            directionImage[xyz+X*nxyz] /= navg;
            directionImage[xyz+Y*nxyz] /= navg;
            directionImage[xyz+Z*nxyz] /= navg;
        }
        return;
	}

}
