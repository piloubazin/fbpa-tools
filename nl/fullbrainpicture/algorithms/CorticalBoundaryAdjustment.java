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
public class CorticalBoundaryAdjustment {

	// jist containers
	private float[] gwbImage=null;
	private float[] cgbImage=null;
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
	private int iterations=5;
	private int repeats=2;
	private float smoothness = 0.5f;
	private float minthickness=2.0f;
	private String lutdir=null;
	private String connectivity="no";
	
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
	public final void setGwbLevelsetImage(float[] val) { gwbImage = val; }
	public final void setCgbLevelsetImage(float[] val) { cgbImage = val; }
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
	public final void setIterations(int val) { iterations = val; }
	public final void setRepeats(int val) { repeats = val; }
	public final void setSmoothness(float val) { smoothness = val; }
	public final void setMinThickness(float val) { minthickness = val; }
	public final void setConnectivity(String val) { connectivity = val; }
	public final void setTopologyLUTdirectory(String val) { lutdir = val; }
	
	// create outputs
	public final float[] getGwbLevelsetImage() { return gwbImage; }
	public final float[] getCgbLevelsetImage() { return cgbImage; }
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
	    
		// labeling, assuming gwb inside cgb
        float[] output = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (gwbImage[xyz]<0 && cgbImage[xyz]<0) output[xyz] = 2.0f;
            else if (cgbImage[xyz]<0) output[xyz] = 1.0f;
            else output[xyz] = 0.0f;
        }
        
        for (int t=0;t<repeats;t++) {
            
            System.out.println("repeat "+(t+1));
            
            // Start with gwb, push boundary inward from cgb
            float[] gwb0 = new float[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                gwb0[xyz] = Numerics.max(gwbImage[xyz], cgbImage[xyz]+minthickness);
            }
            adjustLevelset(gwbImage, gwb0);
		
            // Run the adjustment for gwb
            float[] gwbN = fitBasicBoundarySigmoid(gwbImage, iterations);
            adjustLevelset(gwbImage, gwbN);
            
            // Push boundary outward from gwb
            float[] cgb0 = new float[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                cgb0[xyz] = Numerics.min(cgbImage[xyz], gwbImage[xyz]-minthickness);
            }
            adjustLevelset(cgbImage, cgb0);
		
            // Run the adjustment for gwb
            float[] cgbN = fitBasicBoundarySigmoid(cgbImage, iterations);
            adjustLevelset(cgbImage, cgbN);
            
            // Repeat
		}
		
        for (int xyz=0;xyz<nxyz;xyz++) {
                 if (gwbImage[xyz]<0 && output[xyz]==2.0f) output[xyz] = 8.0f;
            else if (gwbImage[xyz]<0 && output[xyz]==1.0f) output[xyz] = 7.0f;
            else if (gwbImage[xyz]<0 && output[xyz]==0.0f) output[xyz] = 6.0f;
            else if (cgbImage[xyz]<0 && output[xyz]==2.0f) output[xyz] = 5.0f;
            else if (cgbImage[xyz]<0 && output[xyz]==1.0f) output[xyz] = 4.0f;
            else if (cgbImage[xyz]<0 && output[xyz]==0.0f) output[xyz] = 3.0f;
            else if (output[xyz]==2.0f) output[xyz] = 2.0f;
            else if (output[xyz]==1.0f) output[xyz] = 1.0f;
            else output[xyz] = 0.0f;
        }
        
        for (int xyz=0;xyz<nxyz;xyz++) {
            gwbImage[xyz] = Numerics.bounded(gwbImage[xyz],-distance-minthickness, distance+minthickness);
            cgbImage[xyz] = Numerics.bounded(cgbImage[xyz],-distance-minthickness, distance+minthickness);
        }
        
        probaImage = output;
	    return;
	}
	
	public float[] fitBasicBoundarySigmoid(float[] levelset, int iter) {

	    float delta = 0.001f;
	    float dist0 = 2.0f;
	    
	    int dist = Numerics.ceil(distance);
	    
	    // go over voxels close enough to the boundary
	    float[] newlevel = new float[nxyz];
	    float[] oldlevel = new float[nxyz];
	    
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        newlevel[xyz] = levelset[xyz];
	    }

	    for (int t=0;t<iter;t++) {
	        System.out.println("iteration "+(t+1));
            for (int xyz=0;xyz<nxyz;xyz++) {
                oldlevel[xyz] = newlevel[xyz];
            }
            float maxdiff = 0.0f;
            for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
                int xyz = x+nx*y+nx*ny*z;
                
                if (mask[xyz] && Numerics.abs(oldlevel[xyz])<spread) {
                    
                    // grow region
                    float interior = 0.0f;
                    float incount = 0.0f;
                    float exterior = 0.0f;
                    float excount = 0.0f;
                    for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                        int dxyz = dx+nx*dy+nx*ny*dz;
                        if (mask[dxyz]) {
                            if ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImage[dxyz]>contrastImage[xyz])
                                || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) ) {
                                float win = - Numerics.bounded(oldlevel[dxyz]/dist0, -1.0f, 1.0f);
                                if (win<0) {
                                    exterior += win*win*contrastImage[dxyz];
                                    excount += win*win;
                                }
                                if (win>0) {
                                    interior += win*win*contrastImage[dxyz];
                                    incount += win*win;
                                }
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
                        
                            // offset as the location where probability in/out transition?
                            float inbound = 0.0f;
                            float exbound = 0.0f;
                            incount = 0.0f;
                            excount = 0.0f;
                            float bdcount = 0.0f;
                            for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                                int dxyz = dx+nx*dy+nx*ny*dz;
                                if (mask[dxyz]) {
                                    if ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImage[dxyz]>contrastImage[xyz]) 
                                        || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) ) {
                                        float wgtin = Numerics.bounded((contrastImage[dxyz]-interior)/(exterior-interior), delta, 1.0f-delta);
                                        float wgtex = Numerics.bounded((exterior-contrastImage[dxyz])/(exterior-interior), delta, 1.0f-delta);
                                        
                                        float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(oldlevel[dxyz]/dist0)), delta, 1.0f-delta);
                        
                                        inbound += oldlevel[dxyz]*wgtx*wgtin;
                                        incount += wgtx*wgtin;
                        
                                        exbound += oldlevel[dxyz]*wgtx*wgtex;
                                        excount += wgtx*wgtex;
                                        
                                        bdcount += wgtx;
                                    }
                                }
                            }
                            // seems to be a good compromise, using the relative probabilities for in/out as spatial bias
                            if (bdcount>0) {
                                float offset = 0.5f*(inbound/bdcount + exbound/bdcount)/(incount/bdcount + excount/bdcount);
                                newlevel[xyz] = levelset[xyz]-offset;
                                
                                maxdiff = Numerics.max(maxdiff,Numerics.abs(offset));
                            }
                        }
                    }
                }
            }
            System.out.println(" max difference: "+maxdiff);
        }
        return newlevel;
    }

    void adjustLevelset(float[] source, float[] target) {
        
        Gdm3d gdm = new Gdm3d(source, distance+2.0f, nx, ny, nz, rx, ry, rz,
						null, null, target, 0.0f, 0.0f, smoothness, 1.0f-smoothness,
						connectivity, lutdir);
		
		gdm.evolveNarrowBand(50, 0.01f);
		
		source = gdm.getLevelSet();
    }
    
}
