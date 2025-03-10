package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import java.util.*;

/*
 * @author Pierre-Louis Bazin
 */
public class LevelsetBoundaryAdjustment {

	// jist containers
	private float[] levelsetImage=null;
	private float[] contrastImage=null;
	private int[] maskImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	
	private float distance;
	private float spread;
	private byte contrastType;
	private static final	byte	INCREASING = 1;
	private static final	byte	DECREASING = -1;
	private static final	byte	BOTH = 0;
	
	private float[] probaImage;
	
	// intermediate results
	private boolean[] mask;
	
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

	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;

	// create inputs
	public final void setLevelsetImage(float[] val) { levelsetImage = val; }
	public final void setContrastImage(float[] val) { contrastImage = val; }
	public final void setMaskImage(int[] val) { maskImage = val; }
	    
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	public final void setBoundaryDistance(float val) { distance = val; }
	public final void setLocalSpread(float val) { spread = val; }
	public final void setContrastType(String val) {
	    if (val.equals("increasing")) contrastType = INCREASING;
	    else if (val.equals("decreasing")) contrastType = DECREASING;
	    else contrastType = BOTH;; 
	}
	
	// create outputs
	public final float[] getLevelsetImage() { return levelsetImage; }
	public final float[] getProbaImage() { return probaImage; }
	
	public void execute(){
	    
	    // make mask
	    mask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mask[xyz] = true;
		    if (contrastImage[xyz]==0) mask[xyz] = false;
		    if (maskImage!=null && maskImage[xyz]==0) mask[xyz] = false;
		}
		maskImage = null;
	    
		fitBasicBoundarySigmoid();
	}
	
	public void fitBasicBoundarySigmoid() {

	    float delta = 0.001f;
	    float delta0 = 0.1f;
	    float dist0 = 2.0f;
	    //int nngb = 30;
	    
	    int dist = Numerics.ceil(distance);
	    
	    // go over voxels close enough to the boundary
	    float[] newlevel = new float[nxyz];
	    float[] newcount = new float[nxyz];

	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
	        int xyz = x+nx*y+nx*ny*z;
	        
	        newlevel[xyz] = levelsetImage[xyz];
	        
	        if (mask[xyz] && Numerics.abs(levelsetImage[xyz])<spread) {
	            
	            // grow region
	            float interior = 0.0f;
	            float incount = 0.0f;
	            float exterior = 0.0f;
	            float excount = 0.0f;
	            for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
	                int dxyz = dx+nx*dy+nx*ny*dz;
	                if (mask[dxyz]) {
	                    float win = - Numerics.bounded(levelsetImage[dxyz]/dist0, -1.0f, 1.0f);
	                    if (win>0) {
                            interior += win*win*contrastImage[dxyz];
                            incount += win*win;
                        } else {
                            exterior += win*win*contrastImage[dxyz];
                            excount += win*win;
                        }
                    }
                }
                // skip if one is empty
                if (incount>0 && excount>0) {
                    interior /= incount;
                    exterior /= excount;
                    
                    // only take into account correct contrast values
                    if ( (contrastType==INCREASING && exterior>interior) 
                        || (contrastType==DECREASING && exterior<interior)
                        || (contrastType==BOTH) ) {
                    
                        // offset
                        float offset = 0.0f;
                        float offcount = 0.0f;
                        for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                            int dxyz = dx+nx*dy+nx*ny*dz;
                            if (mask[dxyz]) {
                                float wgty = Numerics.bounded((contrastImage[dxyz]-interior)/(exterior-interior), delta, 1.0f-delta)
                                            *Numerics.bounded((exterior-contrastImage[dxyz])/(exterior-interior), delta, 1.0f-delta);
                                float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelsetImage[dxyz]/dist0)), delta, 1.0f-delta);
                
                                offset += levelsetImage[dxyz]*wgtx*wgty;
                                offcount += wgtx*wgty;
                            }
                        }
                        if (offcount>0) {
                            offset /= offcount;
                        
                            // propagate the offset over the levelset values, with weights
                            /* maybe too smooth that way?
                            for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                                int dxyz = dx+nx*dy+nx*ny*dz;
                                if (mask[dxyz]) {
                                    float wsample = Numerics.bounded( 1.0f - ((dx-x)*(dx-x) + (dy-y)*(dy-y) +(dz-z)*(dz-z))/(distance*distance), 0.0f, 1.0f);
                                    newlevel[dxyz] += wsample*(levelsetImage[dxyz]-offset);
                                    newcount[dxyz] += wsample;
                                }
                            }*/
                            newlevel[xyz] = levelsetImage[xyz]-offset;
                        }
                    }
                }
            }
        }
        for (int xyz=0;xyz<nxyz;xyz++) if (newcount[xyz]>0) {
            newlevel[xyz] /= newcount[xyz];
        }
        float[] output = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (levelsetImage[xyz]<0 && newlevel[xyz]<0) output[xyz] = 2.0f;
            if (levelsetImage[xyz]<0 && newlevel[xyz]>0) output[xyz] = 1.0f;
            if (levelsetImage[xyz]>0 && newlevel[xyz]<0) output[xyz] = 3.0f;
            if (levelsetImage[xyz]>0 && newlevel[xyz]>0) output[xyz] = 0.0f;
        }
        levelsetImage = output;
        //levelsetImage = newlevel;
        //probaImage = newcount;
        probaImage = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            probaImage[xyz] = levelsetImage[xyz]-newlevel[xyz];
        }
	    return;
	}

}
