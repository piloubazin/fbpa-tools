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
	private float[][] contrastImages=null;
	private int[] maskImage=null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	private int nc;
	
	private float distance;
	private float spread;
	private byte[] gwbContrastTypes;
	private byte[] cgbContrastTypes;
	private static final	byte	INCREASING = 1;
	private static final	byte	DECREASING = -1;
	private static final	byte	BOTH = 0;
	private int iterations=5;
	private int repeats=10;
	private int pairs=2;
	private float smoothness = 0.5f;
	private float minthickness=2.0f;
	private float maskthickness=1.0f;
	private float gwboffset=2.0f;
	private float cgboffset=2.0f;
	private String lutdir=null;
	private String connectivity="no";
	private float sampleRatio = 0.01f;
	private float lvlRatio = 0.5f;
	
	// supervoxel stuff
	int nsx,nsy,nsz,nsxyz;
	float noiseRatio = 0.1f;
	
	private float[] probaImage;
	
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
	public final void setContrastNumber(int val) {
	    nc = val;
	    contrastImages = new float[nc][];
	    gwbContrastTypes = new byte[nc];
	    cgbContrastTypes = new byte[nc];
	}
	public final void setContrastImageAt(int c, float[] val) { contrastImages[c] = val; }
	public final void setMaskImage(int[] val) { maskImage = val; }
	    
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	public final void setBoundaryDistance(float val) { distance = val; }
	public final void setLocalSpread(float val) { spread = val; }
	public final void setGwbContrastTypeAt(int c, String val) {
	    if (val.equals("increasing")) gwbContrastTypes[c] = INCREASING;
	    else if (val.equals("decreasing")) gwbContrastTypes[c] = DECREASING;
	    else gwbContrastTypes[c] = BOTH;; 
	}
	public final void setCgbContrastTypeAt(int c, String val) {
	    if (val.equals("increasing")) cgbContrastTypes[c] = INCREASING;
	    else if (val.equals("decreasing")) cgbContrastTypes[c] = DECREASING;
	    else cgbContrastTypes[c] = BOTH;; 
	}
	public final void setIterations(int val) { iterations = val; }
	public final void setRepeats(int val) { repeats = val; }
	public final void setPairs(int val) { pairs = val; }
	public final void setSmoothness(float val) { smoothness = val; }
	public final void setMinThickness(float val) { minthickness = val; }
	public final void setGwbOffset(float val) { gwboffset = val; }
	public final void setCgbOffset(float val) { cgboffset = val; }
	public final void setConnectivity(String val) { connectivity = val; }
	public final void setTopologyLUTdirectory(String val) { lutdir = val; }
	public final void setNoiseRatio(float val) { noiseRatio = val; }
	
	// create outputs
	public final float[] getGwbLevelsetImage() { return gwbImage; }
	public final float[] getCgbLevelsetImage() { return cgbImage; }
	public final float[] getProbaImage() { return probaImage; }
	
	public void executeLocal(){
	    
	    // make mask
	    boolean[] mainmask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mainmask[xyz] = true;
		    for (int c=0;c<nc;c++) {
		        if (contrastImages[c][xyz]==0) mainmask[xyz] = false;
		    }
		    if (maskImage!=null && maskImage[xyz]==0) mainmask[xyz] = false;
		}
		maskImage = null;
	    
        // Start with gwb, push boundary inward from cgb
        // Push cgb boundary outward from gwb
        float[] lvl1 = new float[nxyz];
        float[] lvl2 = new float[nxyz];
        boolean[] gwbmask = new boolean[nxyz];
        boolean[] cgbmask = new boolean[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            lvl1[xyz] = Numerics.max(gwbImage[xyz]+gwboffset, cgbImage[xyz]+minthickness);
            lvl2[xyz] = Numerics.min(cgbImage[xyz]+cgboffset, gwbImage[xyz]-minthickness);
            gwbmask[xyz] = mainmask[xyz];
            if (cgbImage[xyz]>-maskthickness) gwbmask[xyz] = false;
            cgbmask[xyz] = mainmask[xyz];
            if (gwbImage[xyz]<maskthickness) cgbmask[xyz] = false;
            // direct copy for debug
            gwbImage[xyz] = lvl1[xyz];
            cgbImage[xyz] = lvl2[xyz];
        }
        //gwbImage = adjustLevelset(gwbImage, lvl1);
        //cgbImage = adjustLevelset(cgbImage, lvl2);
		
		// labeling, assuming gwb inside cgb
        float[] output = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (gwbImage[xyz]<0 && cgbImage[xyz]<0) output[xyz] = 2.0f;
            else if (cgbImage[xyz]<0) output[xyz] = 1.0f;
            else output[xyz] = 0.0f;
        }

        for (int p=0;p<pairs;p++) {

            System.out.println("pair "+(p+1)+": GWB");
                
            for (int xyz=0;xyz<nxyz;xyz++) {
                gwbmask[xyz] = mainmask[xyz];
                if (cgbImage[xyz]>-maskthickness) gwbmask[xyz] = false;
            }            
            for (int t=0;t<repeats;t++) {
                
                System.out.println("repeat "+(t+1));
                
                // Run the adjustment for gwb
                lvl1 = fitJointBoundarySigmoid(gwbImage, iterations, gwbContrastTypes, gwbmask);
                gwbImage = adjustLevelset(gwbImage, lvl1);
            }
            
            System.out.println("pair "+(p+1)+": CGB");
            
            for (int xyz=0;xyz<nxyz;xyz++) {
                cgbmask[xyz] = mainmask[xyz];
                if (gwbImage[xyz]<maskthickness) cgbmask[xyz] = false;
            }                
            for (int t=0;t<repeats;t++) {
                
                System.out.println("repeat "+(t+1));
            
                // Run the adjustment for cgb
                lvl2 = fitJointBoundarySigmoid(cgbImage, iterations, cgbContrastTypes, cgbmask);
                cgbImage = adjustLevelset(cgbImage, lvl2);
            }
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
	
	public void executeSuper(){
	    
	    // make mask
	    boolean[] mainmask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mainmask[xyz] = true;
		    for (int c=0;c<nc;c++) {
		        if (contrastImages[c][xyz]==0) mainmask[xyz] = false;
		    }
		    if (maskImage!=null && maskImage[xyz]==0) mainmask[xyz] = false;
		}
		maskImage = null;
		
		// labeling, assuming gwb inside cgb
        float[] output = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (gwbImage[xyz]<0 && cgbImage[xyz]<0) output[xyz] = 2.0f;
            else if (cgbImage[xyz]<0) output[xyz] = 1.0f;
            else output[xyz] = 0.0f;
        }

        float[] lvl = new float[nxyz];
        for (int p=0;p<pairs;p++) {

            System.out.println("pair "+(p+1)+": GWB");
                
            // Start with gwb, push boundary inward from cgb
            boolean[] gwbmask = new boolean[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                lvl[xyz] = Numerics.max(gwbImage[xyz], cgbImage[xyz]+minthickness);
                gwbmask[xyz] = mainmask[xyz];
                if (cgbImage[xyz]>-maskthickness) gwbmask[xyz] = false;
            }
            gwbImage = adjustLevelset(gwbImage, lvl);
            
            // precompute super-voxel parcels (different for each mask)
            int[] parcel = supervoxelParcellation(contrastImages[0], gwbmask, Numerics.ceil(distance), noiseRatio);
	    
            for (int t=0;t<repeats;t++) {
                
                System.out.println("repeat "+(t+1));
                
                // Run the adjustment for gwb
                lvl = fitSupervoxelBoundarySigmoid(gwbImage, parcel, iterations, gwbContrastTypes, gwbmask);
                gwbImage = adjustLevelset(gwbImage, lvl);
            }
            
            System.out.println("pair "+(p+1)+": CGB");
            
            // Push cgb boundary outward from gwb
            boolean[] cgbmask = new boolean[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                lvl[xyz] = Numerics.min(cgbImage[xyz], gwbImage[xyz]-minthickness);
                cgbmask[xyz] = mainmask[xyz];
                if (gwbImage[xyz]<maskthickness) cgbmask[xyz] = false;
            }
            cgbImage = adjustLevelset(cgbImage, lvl);
            
            // precompute super-voxel parcels (different for each mask)
            parcel = supervoxelParcellation(contrastImages[0], cgbmask, Numerics.ceil(distance), noiseRatio);

            for (int t=0;t<repeats;t++) {
                
                System.out.println("repeat "+(t+1));
            
                // Run the adjustment for cgb
                lvl = fitSupervoxelBoundarySigmoid(cgbImage, parcel, iterations, cgbContrastTypes, cgbmask);
                cgbImage = adjustLevelset(cgbImage, lvl);
            }
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
	
	public float[] fitBasicBoundarySigmoid(float[] levelset, int iter, byte[] contrastTypes, boolean[] mask) {

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
                    float[] interior = new float[nc];
                    float[] incount = new float[nc];
                    float[] exterior = new float[nc];
                    float[] excount = new float[nc];
                    for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                        int dxyz = dx+nx*dy+nx*ny*dz;
                        if (mask[dxyz]) {
                            for (int c=0;c<nc;c++) {
                                if ( (contrastTypes[c]==INCREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                               || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                    || (contrastTypes[c]==DECREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                                || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                                    || (contrastTypes[c]==BOTH) ) {
    
                                    float win = - Numerics.bounded(oldlevel[dxyz]/dist0, -1.0f, 1.0f);
                                    if (win<0) {
                                        exterior[c] += win*win*contrastImages[c][dxyz];
                                        excount[c] += win*win;
                                    }
                                    if (win>0) {
                                        interior[c] += win*win*contrastImages[c][dxyz];
                                        incount[c] += win*win;
                                    }
                                } 
                            }
                        }
                    }
                    float inbound = 0.0f;
                    float exbound = 0.0f;
                    float bdcount = 0.0f;
                    float inwgt = 0.0f;
                    float exwgt = 0.0f;
                    for (int c=0;c<nc;c++) {
                        // skip if one is empty
                        if (incount[c]>0 && excount[c]>0) {
                            interior[c] /= incount[c];
                            exterior[c] /= excount[c];
                            
                            // only take into account correct contrast values
                            if ( (contrastTypes[c]==INCREASING && exterior[c]>interior[c]) 
                                || (contrastTypes[c]==DECREASING && exterior[c]<interior[c])
                                || (contrastTypes[c]==BOTH) ) {
                            
                                // offset as the location where probability in/out transition?
                                for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                                    int dxyz = dx+nx*dy+nx*ny*dz;
                                    if (mask[dxyz]) {
                                        if ( (contrastTypes[c]==INCREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                                       || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                            || (contrastTypes[c]==DECREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                                        || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                                            || (contrastTypes[c]==BOTH) ) {
                                        
                                            // to check??
                                            float wgtin = Numerics.bounded((contrastImages[c][dxyz]-interior[c])/(exterior[c]-interior[c]), delta, 1.0f-delta);
                                            float wgtex = Numerics.bounded((exterior[c]-contrastImages[c][dxyz])/(exterior[c]-interior[c]), delta, 1.0f-delta);
                                            
                                            float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(oldlevel[dxyz]/dist0)), delta, 1.0f-delta);
                            
                                            inbound += oldlevel[dxyz]*wgtx*wgtin;
                                            inwgt += wgtx*wgtin;
                            
                                            exbound += oldlevel[dxyz]*wgtx*wgtex;
                                            exwgt += wgtx*wgtex;
                                            
                                            bdcount += wgtx;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // seems to be a good compromise, using the relative probabilities for in/out as spatial bias
                    if (bdcount>0) {
                        float offset = 0.5f*(inbound/bdcount + exbound/bdcount)/(inwgt/bdcount + exwgt/bdcount);
                        newlevel[xyz] = levelset[xyz]-offset;
                        
                        maxdiff = Numerics.max(maxdiff,Numerics.abs(offset));
                    }
                }
            }
            System.out.println(" max difference: "+maxdiff);
        }
        return newlevel;
    }

	public float[] fitJointBoundarySigmoid(float[] levelset, int iter, byte[] contrastTypes, boolean[] mask) {

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
                    float[] interior = new float[nc];
                    float[] incount = new float[nc];
                    float[] exterior = new float[nc];
                    float[] excount = new float[nc];
                    for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                        int dxyz = dx+nx*dy+nx*ny*dz;
                        if (mask[dxyz]) {
                            for (int c=0;c<nc;c++) {
                                if ( (contrastTypes[c]==INCREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                               || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                    || (contrastTypes[c]==DECREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                                || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                                    || (contrastTypes[c]==BOTH) ) {
    
                                    float win = - Numerics.bounded(oldlevel[dxyz]/dist0, -1.0f, 1.0f);
                                    if (win<0) {
                                        exterior[c] += win*win*contrastImages[c][dxyz];
                                        excount[c] += win*win;
                                    }
                                    if (win>0) {
                                        interior[c] += win*win*contrastImages[c][dxyz];
                                        incount[c] += win*win;
                                    }
                                } 
                            }
                        }
                    }
                    float inbound = 0.0f;
                    float exbound = 0.0f;
                    float bdcount = 0.0f;
                    float inwgt = 0.0f;
                    float exwgt = 0.0f;
                    // offset as the location where probability in/out transition?
                    for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                        int dxyz = dx+nx*dy+nx*ny*dz;
                        if (mask[dxyz]) {
                            float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(oldlevel[dxyz]/dist0)), delta, 1.0f-delta);
                            
                            float wgtin=1.0f;
                            float wgtex=1.0f;
                            int count=0;
                            for (int c=0;c<nc;c++) {
                                // skip if one is empty
                                if (incount[c]>0 && excount[c]>0) {
                                    interior[c] /= incount[c];
                                    exterior[c] /= excount[c];
                                    
                                    // only take into account correct contrast values
                                    if ( (contrastTypes[c]==INCREASING && exterior[c]>interior[c]) 
                                        || (contrastTypes[c]==DECREASING && exterior[c]<interior[c])
                                        || (contrastTypes[c]==BOTH) ) {
                
                                        if ( (contrastTypes[c]==INCREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                                       || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                            || (contrastTypes[c]==DECREASING && ( (oldlevel[dxyz]>oldlevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                                        || (oldlevel[dxyz]<=oldlevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                                            || (contrastTypes[c]==BOTH) ) {
                            
                                            // to check??
                                            wgtin *= Numerics.bounded((contrastImages[c][dxyz]-interior[c])/(exterior[c]-interior[c]), delta, 1.0f-delta);
                                            wgtex *= Numerics.bounded((exterior[c]-contrastImages[c][dxyz])/(exterior[c]-interior[c]), delta, 1.0f-delta);
                                            count++;
                                        }
                                    }
                                }
                            }
                            if (count>0) {
                
                                inbound += oldlevel[dxyz]*wgtx*wgtin;
                                inwgt += wgtx*wgtin;
                
                                exbound += oldlevel[dxyz]*wgtx*wgtex;
                                exwgt += wgtx*wgtex;
                                
                                bdcount += wgtx;
                            }
                        }
                    }
                    // seems to be a good compromise, using the relative probabilities for in/out as spatial bias
                    if (bdcount>0) {
                        float offset = 0.5f*(inbound/bdcount + exbound/bdcount)/(inwgt/bdcount + exwgt/bdcount);
                        newlevel[xyz] = levelset[xyz]-offset;
                        
                        maxdiff = Numerics.max(maxdiff,Numerics.abs(offset));
                    }
                }
            }
            System.out.println(" max difference: "+maxdiff);
        }
        return newlevel;
    }

	public float[] fitFasterBoundarySigmoid(float[] levelset, int iter, byte[] contrastTypes, boolean[] mask) {

	    float delta = 0.001f;
	    //float dist0 = 1.0f;
	    float dist0 = distance/3.0f;
	    
	    int dist = Numerics.ceil(distance);
	    
	    // ratio x approx half of the search volume (removing boundary voxels)
	    float mincount = sampleRatio*4.0f*dist*dist*dist;
	    
	    // build smaller arrays for speed
	    int nspread = 0;
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz] && Numerics.abs(levelset[xyz])<spread) nspread++;
	    
	    // go over voxels close enough to the boundary
	    float[] offlevel = new float[nxyz];
	    float[] offlist = new float[nspread];
	    int[] coord = new int[nspread];
	    float[][] interior = new float[nc][nspread];
	    float[][] exterior = new float[nc][nspread];
	    float[][] incount = new float[nc][nspread];
	    float[][] excount = new float[nc][nspread];
	    boolean[] changed = new boolean[nxyz];
	    
	    int s=0;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        //offlevel[xyz] = 0.0f;
	        if (mask[xyz] && Numerics.abs(levelset[xyz])<spread) {
                //offlist[s] = 0.0f;
                coord[s] = xyz;
                s++;
            }
	    }
	    // precompute region interior, exterior?
	    for (s=0;s<nspread;s++) {
            for (int c=0;c<nc;c++) {
                incount[c][s] = 0.0f;
                excount[c][s] = 0.0f;
                int xyz = coord[s];
                float lvl = levelset[xyz] - offlevel[xyz];
                for (int dx=-dist;dx<=dist;dx++) for (int dy=-dist;dy<=dist;dy++) for (int dz=-dist;dz<=dist;dz++) {
                    int dxyz = xyz + dx + nx*dy + nx*ny*dz;
                    if (mask[dxyz]) {
                        float dlvl = levelset[dxyz] - offlevel[dxyz];
                        // here we exclude the center
                        if ( (contrastTypes[c]==INCREASING && ( (dlvl>lvl && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                           || (dlvl<lvl && contrastImages[c][dxyz]<contrastImages[c][xyz]) ) ) 
                                || (contrastTypes[c]==DECREASING && ( (dlvl>lvl && contrastImages[c][dxyz]<contrastImages[c][xyz]) 
                                                            || (dlvl<lvl && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                                || (contrastTypes[c]==BOTH) ) {
                                
                            float win = - Numerics.bounded(dlvl/dist0, -1.0f, 1.0f);
                            if (win<0) {
                                exterior[c][s] += win*win*contrastImages[c][dxyz];
                                excount[c][s] += win*win;
                            }
                            if (win>0) {
                                interior[c][s] += win*win*contrastImages[c][dxyz];
                                incount[c][s] += win*win;
                            }
                        }
                    }
                }
                // skip if one is empty
                if (incount[c][s]>0 && excount[c][s]>0) {
                    interior[c][s] /= incount[c][s];
                    exterior[c][s] /= excount[c][s];
                }
            }
        }

	    // multiple iterations of the offset within the same basis
	    for (int t=0;t<iter;t++) {
	        for (s=0;s<nspread;s++) {
                offlevel[coord[s]] = offlist[s];
            }
            // do we spread results to neighboring regions??
            for (s=0;s<nspread;s++) if (!changed[coord[s]]) {
                float ngboff = 0.0f;
                float ngb=0;
                for (int dx=-dist;dx<=dist;dx++) for (int dy=-dist;dy<=dist;dy++) for (int dz=-dist;dz<=dist;dz++) {
                    int dxyz = coord[s] + dx + nx*dy + nx*ny*dz;
                    if (changed[dxyz]) {
                        float wgt = (float)FastMath.exp(-0.5*(dx*dx+dy*dy+dz*dz)/(dist0*dist0));
                        ngboff += wgt*offlevel[dxyz];
                        ngb += wgt;
                    }
                }
                if (ngb>0) {
                    offlevel[coord[s]] = ngboff/ngb;
                }
            }
            int ninvalid=0;
            float maxdiff = 0.0f;
            for (s=0;s<nspread;s++) {
                int xyz = coord[s];
                
                float inbound = 0.0f;
                float exbound = 0.0f;
                float insum = 0.0f;
                float exsum = 0.0f;
                float bdsum = 0.0f;
                
                boolean valid=true;
                for (int c=0;c<nc;c++) {

                    // only take into account correct contrast values
                    if ( (contrastTypes[c]==INCREASING && exterior[c][s]<interior[c][s]) 
                      || (contrastTypes[c]==DECREASING && exterior[c][s]>interior[c][s]) ) {
                        valid=false;
                    }
                    // and make sure there's enough samples
                    if (incount[c][s]<=mincount || excount[c][s]<=mincount) {
                        valid = false;
                    }
                }
                //valid = true;
                if (!valid) {
                    ninvalid++;
                    changed[coord[s]] = false;
                } else {    
                    float lvl = levelset[xyz] - offlevel[xyz];
                    for (int dx=-dist;dx<=dist;dx++) for (int dy=-dist;dy<=dist;dy++) for (int dz=-dist;dz<=dist;dz++) {
                        int dxyz = xyz + dx + nx*dy + nx*ny*dz;
                        if (mask[dxyz]) {
                            float dlvl = levelset[dxyz] - offlevel[dxyz];
                            //float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(reslevel[dxyz]/dist0)), delta, 1.0f-delta);
                            float wgtx = (float)FastMath.exp(-0.5*Numerics.square(dlvl/dist0));
                            float wgtin = 0.0f;
                            float wgtex = 0.0f;
                            for (int c=0;c<nc;c++) {
                                // here we include the center (to avoid being pulled by outliers)
                                if ( (contrastTypes[c]==INCREASING && ( (dlvl>=lvl && contrastImages[c][dxyz]>=contrastImages[c][xyz]) 
                                                                   || (dlvl<=lvl && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                        || (contrastTypes[c]==DECREASING && ( (dlvl>=lvl && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                                    || (dlvl<=lvl && contrastImages[c][dxyz]>=contrastImages[c][xyz]) ) ) 
                                        || (contrastTypes[c]==BOTH) ) {
                                            
                                    // oreder not important, symmetric values
                                    wgtin += Numerics.bounded((exterior[c][s]-contrastImages[c][dxyz])/(exterior[c][s]-interior[c][s]), delta, 1.0f-delta);
                                    wgtex += Numerics.bounded((contrastImages[c][dxyz]-interior[c][s])/(exterior[c][s]-interior[c][s]), delta, 1.0f-delta);
                                 }
                            }
                            //inbound += incount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtin;
                            //inbound += excount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtin;
                            inbound += dlvl*wgtx*wgtin;
                            insum += wgtx*wgtin;
                    
                            //exbound += excount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtex;
                            //exbound += incount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtex;
                            exbound += dlvl*wgtx*wgtex;
                            exsum += wgtx*wgtex;
                                    
                            bdsum += wgtx;
                        }
                    }
                    // seems to be a good compromise, using the relative probabilities for in/out as spatial bias
                    if (bdsum>0 && insum>0 && exsum>0) {
                        float offset = 0.5f*(inbound/bdsum + exbound/bdsum)/(insum/bdsum + exsum/bdsum);
                        
                        offlist[s] += offset;
                        changed[coord[s]] = true;
                        
                        maxdiff = Numerics.max(maxdiff,Numerics.abs(offset));
                    } else {
                        System.out.print("!");
                        ninvalid++;
                        changed[coord[s]] = false;
                    }
                }
            }
            if (t==0) System.out.println("ratio invalid: "+((float)ninvalid/(float)nspread));
            System.out.println("iteration "+(t+1)+" max difference: "+maxdiff);
        }
        
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        offlevel[xyz] = levelset[xyz] - offlevel[xyz];
        }
        return offlevel;
    }

    float[] adjustLevelset(float[] source, float[] target) {
        
        Gdm3d gdm = new Gdm3d(source, distance+2.0f, nx, ny, nz, rx, ry, rz,
						null, null, target, 0.0f, 0.0f, smoothness, 1.0f-smoothness,
						connectivity, lutdir);
		
		gdm.evolveNarrowBand(50, 0.01f);
		
		return gdm.getLevelSet();
    }

    int[] supervoxelParcellation(float[] image, boolean[] mask, int scaling, float noise) {
	    // Compute the supervoxel grid
	    System.out.println("original dimensions: ("+nx+", "+ny+", "+nz+")");
	    nsx = Numerics.floor(nx/scaling);
	    nsy = Numerics.floor(ny/scaling);
	    nsz = Numerics.floor(nz/scaling);
	    nsxyz = nsx*nsy*nsz;
	    System.out.println("rescaled dimensions: ("+nsx+", "+nsy+", "+nsz+")");
	    
	    // init downscaled images
	    int[] parcel = new int[nxyz];
	    float[] rescaled = new float[nsxyz];
	    //memsImage = new float[nsxyz];
	    int[] count = new int[nsxyz];
	    
	    // init supervoxel centroids
	    // include all supervoxels with non-zero values inside
		float[][] centroid = new float[3][nsxyz];
	    for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
	        int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        centroid[X][xyzs] = 0.0f;
	        centroid[Y][xyzs] = 0.0f;
	        centroid[Z][xyzs] = 0.0f;
	        count[xyzs] = 0;
	        for (int dx=0;dx<scaling;dx++) for (int dy=0;dy<scaling;dy++) for (int dz=0;dz<scaling;dz++) {
	            int xyz = Numerics.floor(xs*scaling)+dx+nx*(Numerics.floor(ys*scaling)+dy)+nx*ny*(Numerics.floor(zs*scaling)+dz);
	            if (mask[xyz]) {
                    centroid[X][xyzs] += Numerics.floor(xs*scaling) + dx;
                    centroid[Y][xyzs] += Numerics.floor(ys*scaling) + dy;
                    centroid[Z][xyzs] += Numerics.floor(zs*scaling) + dz;
                    count[xyzs]++;
                }
            }
            if (count[xyzs]>0) {
                centroid[X][xyzs] /= count[xyzs];
                centroid[Y][xyzs] /= count[xyzs];
                centroid[Z][xyzs] /= count[xyzs];
	        }
	    }
	    
	    // init: search for voxel with lowest gradient within the region instead? (TODO)
	    // OR voxel most representative?
	    double[] selection = new double[27];
	    Percentile median = new Percentile();
	    for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
	        int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        int x0 = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y0 = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z0 = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz0 = x0+nx*y0+nx*ny*z0;
	        
	        int s=0;
	        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	            selection[s] = image[xyz0+dx+nx*dy+nx*ny*dz];
	            s++;
	        }
	        double med = median.evaluate(selection, 50.0);    
	        for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	            if (image[xyz0+dx+nx*dy+nx*ny*dz]==med) {
                    centroid[X][xyzs] = x0+dx;
                    centroid[Y][xyzs] = y0+dy;
                    centroid[Z][xyzs] = z0+dz;
                    dx=2;dy=2;dz=2;
	            }
	        }
	    }
	    // Estimate approximate min,max from sampled grid values
	    float Imin = 1e9f;
	    float Imax = -1e9f;
        for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
		    int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        
		    int x = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz = x+nx*y+nx*ny*z;	        
	        if (mask[xyz]) {
	            if (image[xyz]<Imin) Imin = image[xyz];
	            if (image[xyz]>Imax) Imax = image[xyz];
	        }
	    }
	    // normalize the noise parameter by intensity, but not by distance (-> same speed indep of scale)
	    System.out.println("intensity scale: ["+Imin+", "+Imax+"]");
	    if (Imax>Imin) {
	        noise = noise*noise*(Imax-Imin)*(Imax-Imin);
	    }
	    
	    // start a voxel heap at each center
	    BinaryHeap4D heap = new BinaryHeap4D(nx*ny+ny*nz+nz*nx, BinaryHeap4D.MINTREE);
		boolean[] processed = new boolean[nx*ny*nz];
		for (int xs=0;xs<nsx;xs++) for (int ys=0;ys<nsy;ys++) for (int zs=0;zs<nsz;zs++) {
		    int xyzs = xs+nsx*ys+nsx*nsy*zs;
	        count[xyzs]=0;
	        
	        int x = Numerics.bounded(Numerics.floor(centroid[X][xyzs]),1,nx-2);
	        int y = Numerics.bounded(Numerics.floor(centroid[Y][xyzs]),1,ny-2);
	        int z = Numerics.bounded(Numerics.floor(centroid[Z][xyzs]),1,nz-2);
	        int xyz = x+nx*y+nx*ny*z;	        
	        if (mask[xyz]) {
	            // set as starting point
	            parcel[xyz] = xyzs+1;
	            rescaled[xyzs] = image[xyz];
	            count[xyzs] = 1;
	            processed[xyz] = true;
	            
	            // add neighbors to the tree
	            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	                if (dx*dx+dy*dy+dz*dz==1 && x+dx>=0 && y+dy>=0 && z+dz>=0 && x+dx<nx && y+dy<ny && z+dz<nz) {
                        int xyznb = x+dx+nx*(y+dy)+nx*ny*(z+dz);
                         // exclude zero as mask
                        if (mask[xyznb]) {
                        
                            // distance function
                            float dist = (x+dx-centroid[X][xyzs])*(x+dx-centroid[X][xyzs])
                                        +(y+dy-centroid[Y][xyzs])*(y+dy-centroid[Y][xyzs])
                                        +(z+dz-centroid[Z][xyzs])*(z+dz-centroid[Z][xyzs]);
                                    
                            float contrast = (image[xyznb]-rescaled[xyzs])
                                            *(image[xyznb]-rescaled[xyzs]);
                                        
                            heap.addValue(noise*dist+contrast, x+dx,y+dy,z+dz, xyzs+1);
                        }
                    }                    
	            }
	        }
	    }
	    // grow to 
        while (heap.isNotEmpty()) {
        	// extract point with minimum distance
        	float curr = heap.getFirst();
        	int x = heap.getFirstX();
        	int y = heap.getFirstY();
        	int z = heap.getFirstZ();
        	int xyzs = heap.getFirstK()-1;
        	heap.removeFirst();
        	int xyz = x+nx*y+nx*ny*z;
        	
			if (processed[xyz])  continue;
			
        	// update the cluster
			parcel[xyz] = xyzs+1;
            rescaled[xyzs] = count[xyzs]*rescaled[xyzs] + image[xyz];
            
            centroid[X][xyzs] = count[xyzs]*centroid[X][xyzs] + x;
	        centroid[Y][xyzs] = count[xyzs]*centroid[Y][xyzs] + y;
	        centroid[Z][xyzs] = count[xyzs]*centroid[Z][xyzs] + z;
	        
	        count[xyzs] += 1;
	        rescaled[xyzs] /= count[xyzs];
	        centroid[X][xyzs] /= count[xyzs];
	        centroid[Y][xyzs] /= count[xyzs];
	        centroid[Z][xyzs] /= count[xyzs];
	        
	        processed[xyz]=true;
			
            // add neighbors to the tree
            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
	            if (dx*dx+dy*dy+dz*dz==1 && x+dx>=0 && y+dy>=0 && z+dz>=0 && x+dx<nx && y+dy<ny && z+dz<nz) {
                    int xyznb = x+dx+nx*(y+dy)+nx*ny*(z+dz);

                    // exclude zero as mask
                    if (mask[xyznb] && !processed[xyznb]) {
                    
                        // distance function
                        float dist = (x+dx-centroid[X][xyzs])*(x+dx-centroid[X][xyzs])
                                    +(y+dy-centroid[Y][xyzs])*(y+dy-centroid[Y][xyzs])
                                    +(z+dz-centroid[Z][xyzs])*(z+dz-centroid[Z][xyzs]);
                                
                        float contrast = (image[xyznb]-rescaled[xyzs])
                                        *(image[xyznb]-rescaled[xyzs]);
                                        
                        heap.addValue(noise*dist+contrast, x+dx,y+dy,z+dz, xyzs+1);
                    }
                }
            }
		}
		return parcel;
	}

	public float[] fitSupervoxelRestrictedBoundarySigmoid(float[] levelset, int[] parcel, int iter, byte[] contrastTypes, boolean[] mask) {

	    float delta = 0.001f;
	    //float dist0 = 1.0f;
	    float dist0 = distance/3.0f;
	    
	    int dist = Numerics.ceil(distance);
	    
	    // ratio x approx half of the search volume (removing boundary voxels)
	    float mincount = sampleRatio*4.0f*dist*dist*dist;
	    
	    // build smaller arrays for speed
	    int nspread = 0;
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz] && Numerics.abs(levelset[xyz])<spread) nspread++;
	    
	    // go over voxels close enough to the boundary
	    float[] offlevel = new float[nxyz];
	    int[] coord = new int[nspread];
	    float[][] interior = new float[nc][nspread];
	    float[][] exterior = new float[nc][nspread];
	    float[][] incount = new float[nc][nspread];
	    float[][] excount = new float[nc][nspread];
	    boolean[] processed = new boolean[nxyz];
	    
	    int s=0;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        //offlevel[xyz] = 0.0f;
	        if (mask[xyz] && Numerics.abs(levelset[xyz])<spread) {
                //offlist[s] = 0.0f;
                coord[s] = xyz;
                s++;
            }
	    }
	    
	    float maxdiff = 0.0f;
	    for (s=0;s<nspread;s++) {
	        // 1. Define ROI from region growing on the levelset function (use a set volume, record coordinates)
	        int nroi = 8*dist*dist*dist;
	        int[] roi = new int[nroi];
	        BinaryHeap1D heap = new BinaryHeap1D(nx*ny+ny*nz+nz*nx, BinaryHeap1D.MINTREE);
	        heap.addValue(0.0f, coord[s]);
	        
	        int r=0;
	        while (heap.isNotEmpty() && r<nroi) {
	            // extract point with minimum distance
	            float curr = heap.getFirst();
                int xyz = heap.getFirstId();
                heap.removeFirst();
                
                if (processed[xyz]) continue;
                
                // update values
                roi[r] = xyz;
                r++;
                processed[xyz]=true;
                
                // add neighbors to the tree
                for (byte b=0;b<6;b++) {
                    int xyznb = fastMarchingNeighborIndex(b, xyz, nx, ny, nz);
                    if (mask[xyznb] && !processed[xyznb]) {
                        // distance function: anisotropic along the levelset
                        float next = curr + (1.0f-lvlRatio)*Numerics.abs(levelset[xyznb]-levelset[xyz]) + lvlRatio;                                        
                        heap.addValue(next, xyznb);
                    }
                }
            }
            nroi = r;
	        
	        // 2. Create positive and negative regions as before, selecting only correct contrasts
	        int xyz = coord[s];
            float lvl = levelset[xyz] - offlevel[xyz];
            for (int c=0;c<nc;c++) {
                incount[c][s] = 0.0f;
                excount[c][s] = 0.0f;
                for (int n=0;n<nroi;n++) {
                    int dxyz = roi[n];
                    float dlvl = levelset[dxyz] - offlevel[dxyz];
                    
                    // here we exclude the center
                    if ( (contrastTypes[c]==INCREASING && ( (dlvl>lvl && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                         || (dlvl<lvl && contrastImages[c][dxyz]<contrastImages[c][xyz]) ) ) 
                      || (contrastTypes[c]==DECREASING && ( (dlvl>lvl && contrastImages[c][dxyz]<contrastImages[c][xyz]) 
                                                         || (dlvl<lvl && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                      || (contrastTypes[c]==BOTH) ) {
                            
                        float win = - Numerics.bounded(dlvl/dist0, -1.0f, 1.0f);
                        if (win<0) {
                            exterior[c][s] += win*win*contrastImages[c][dxyz];
                            excount[c][s] += win*win;
                        }
                        if (win>0) {
                            interior[c][s] += win*win*contrastImages[c][dxyz];
                            incount[c][s] += win*win;
                        }
                      }
                }
                if (incount[c][s]>0 && excount[c][s]>0) {
                    interior[c][s] /= incount[c][s];
                    exterior[c][s] /= excount[c][s];
                }
            }

	        // 3. Update using memberships close to value (note: we can iterate inside the ROI definition)
	        
	        // multiple iterations of the offset within the same basis
            for (int t=0;t<iter;t++) {
                float inbound=0.0f;
                float exbound=0.0f;
                float insum = 0.0f;
                float exsum = 0.0f;
                float bdsum = 0.0f;
                
	            for (int n=0;n<nroi;n++) {
	                int dxyz = roi[n];
                    float dlvl = levelset[dxyz] - offlevel[dxyz];
                    //float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(reslevel[dxyz]/dist0)), delta, 1.0f-delta);
                    //float wgtx = (float)FastMath.exp(-0.5*Numerics.square(dlvl/dist0));
                    float wgtx = 1.0f;
                    float wgtin = 0.0f;
                    float wgtex = 0.0f;
                    for (int c=0;c<nc;c++) {
                        // here we include the center (to avoid being pulled by outliers)
                        if ( (contrastTypes[c]==INCREASING && ( (dlvl>=lvl && contrastImages[c][dxyz]>=contrastImages[c][xyz]) 
                                                           || (dlvl<=lvl && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                || (contrastTypes[c]==DECREASING && ( (dlvl>=lvl && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                            || (dlvl<=lvl && contrastImages[c][dxyz]>=contrastImages[c][xyz]) ) ) 
                                || (contrastTypes[c]==BOTH) ) {
                                    
                            // oreder not important, symmetric values
                            wgtin += Numerics.bounded((exterior[c][s]-contrastImages[c][dxyz])/(exterior[c][s]-interior[c][s]), delta, 1.0f-delta);
                            wgtex += Numerics.bounded((contrastImages[c][dxyz]-interior[c][s])/(exterior[c][s]-interior[c][s]), delta, 1.0f-delta);
                         }
                    }
                    //inbound += incount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtin;
                    //inbound += excount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtin;
                    inbound += dlvl*wgtx*wgtin;
                    insum += wgtx*wgtin;
            
                    //exbound += excount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtex;
                    //exbound += incount[c][s]/(incount[c][s]+excount[c][s])*dlvl*wgtx*wgtex;
                    exbound += dlvl*wgtx*wgtex;
                    exsum += wgtx*wgtex;
                            
                    bdsum += wgtx;
                }
                // seems to be a good compromise, using the relative probabilities for in/out as spatial bias
                if (bdsum>0 && insum>0 && exsum>0) {
                    float offset = 0.5f*(inbound/bdsum + exbound/bdsum)/(insum/bdsum + exsum/bdsum);
                    
                    offlevel[xyz] += offset;
                    
                    maxdiff = Numerics.max(maxdiff,Numerics.abs(offset));
                }
            }
        }
        System.out.println("max difference: "+maxdiff);
        
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        offlevel[xyz] = levelset[xyz] - offlevel[xyz];
        }
        return offlevel;
    }
    
	public float[] fitSupervoxelBoundarySigmoid(float[] levelset, int[] parcel, int iter, byte[] contrastTypes, boolean[] mask) {

	    float delta = 0.001f;
	    //float dist0 = 1.0f;
	    float dist0 = distance/3.0f;
	    
	    int dist = Numerics.ceil(distance);
	    
	    // ratio x approx half of the search volume (removing boundary voxels)
	    float mincount = sampleRatio*4.0f*dist*dist*dist;
	    
	    // define stats over regions (could be precomputed)
	    float[][] avg = new float[nc][nsxyz];
	    float[][] std = new float[nc][nsxyz];
	    int[] count = new int[nsxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) {
		    if (mask[xyz] && parcel[xyz]>0) {
		        count[parcel[xyz]-1]++;
		        for (int c=0;c<nc;c++) {
		            avg[c][parcel[xyz]-1] += contrastImages[c][xyz];
		        }
		    }
		}
		for (int xyzs=0;xyzs<nsxyz;xyzs++) if (count[xyzs]>0) for (int c=0;c<nc;c++) {
            avg[c][xyzs] /= count[xyzs];
		}
	    for (int xyz=0;xyz<nxyz;xyz++) {
		    if (mask[xyz] && parcel[xyz]>0) {
		        for (int c=0;c<nc;c++) {
		            std[c][parcel[xyz]-1] += (avg[c][parcel[xyz]-1]-contrastImages[c][xyz])*(avg[c][parcel[xyz]-1]-contrastImages[c][xyz]);
		        }
		    }
		}
		for (int xyzs=0;xyzs<nsxyz;xyzs++) if (count[xyzs]>1) for (int c=0;c<nc;c++) {
            std[c][xyzs] = (float)FastMath.sqrt(std[c][xyzs]/(count[xyzs]-1.0f));
		}

		float[] meanlvl = new float[nsxyz];
        boolean[] boundary = new boolean[nsxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (mask[xyz] && parcel[xyz]>0) {
                meanlvl[parcel[xyz]-1] += levelset[xyz];
                if (Numerics.abs(levelset[xyz])<spread) boundary[parcel[xyz]-1] = true;
            }
        }
		for (int xyzs=0;xyzs<nsxyz;xyzs++) if (count[xyzs]>0) {
            meanlvl[xyzs] /= count[xyzs];
        }
        
		float[] offparcel = new float[nsxyz];
		// multiple iterations of the offset within the same basis
	    for (int t=0;t<iter;t++) {
	        // 2. find most likely neighbor on opposite side of boundary
	        float maxdiff = 0.0f;
	        for (int xyzs=0;xyzs<nsxyz;xyzs++) if (boundary[xyzs]) {
	            // rank direct neighbors from intensity distance, levelset value, number of valid points (?)
	            int best=-1;
	            double pbest=0.0;
	            for (byte d=0;d<26;d++) {
	                int ngbs = Ngb.neighborIndex(d, xyzs, nsx, nsy, nsz);
	                
	                if (count[ngbs]>0) {
                        float ldist = (meanlvl[ngbs]+offparcel[ngbs]) - (meanlvl[xyzs]+offparcel[xyzs]);
	                
                        float idist = 0.0f;
                        for (int c=0;c<nc;c++) {
                            if (contrastTypes[c]==INCREASING && ldist*(avg[c][ngbs]-avg[c][xyzs])>0) {
                                idist += Numerics.square((avg[c][ngbs]-avg[c][xyzs])/(std[c][ngbs]+std[c][xyzs]));
                            } else if (contrastTypes[c]==DECREASING && ldist*(avg[c][ngbs]-avg[c][xyzs])<0) {
                                idist += Numerics.square((avg[c][ngbs]-avg[c][xyzs])/(std[c][ngbs]+std[c][xyzs]));
                            }
                        }
                        double pngb = (1.0 - FastMath.exp( -0.5*Numerics.square(ldist/dist0) ))
                                      *(1.0 - FastMath.exp( -0.5*idist/nc));
                                      
                        if (pngb>pbest) {
                            pbest = pngb;
                            best = ngbs;
                        }
                    }
                }
                if (best!=-1) {
                    // 3. compute offset
                    float offset = 0.5f*((meanlvl[xyzs]+offparcel[xyzs])-(meanlvl[best]+offparcel[best]));
                    offparcel[xyzs] += offset;
                    if (Numerics.abs(offset)>maxdiff) maxdiff = Numerics.abs(offset);
                } else {
                    System.out.print(".");
                    //ninvalid++;
                    //changed[xyzs] = false;
                }
            }
            //if (t==0) System.out.println("ratio invalid: "+((float)ninvalid/(float)nspread));
            System.out.println("iteration "+(t+1)+" max difference: "+maxdiff);
        }
        float[] offlevel = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (parcel[xyz]>0) {
	        offlevel[xyz] = levelset[xyz] + offparcel[parcel[xyz]-1];
        }
        return offlevel;
    }   
    
    public static final int fastMarchingNeighborIndex(byte d, int id, int nx, int ny, int nz) {
		switch (d) {
			case 0		: 	return id+1; 		
			case 1		:	return id-1;
			case 2		:	return id+nx;
			case 3		:	return id-nx;
			case 4		:	return id+nx*ny;
			case 5		:	return id-nx*ny;
			default		:	return id;
		}
	}


}
