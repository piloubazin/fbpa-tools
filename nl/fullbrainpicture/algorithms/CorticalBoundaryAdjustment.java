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
		    for (int c=0;c<nc;c++) {
		        if (contrastImages[c][xyz]==0) mask[xyz] = false;
		    }
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

        float[] lvl = new float[nxyz];
        for (int p=0;p<pairs;p++) {

            System.out.println("pair "+(p+1));
                
            // Start with gwb, push boundary inward from cgb
            for (int xyz=0;xyz<nxyz;xyz++) {
                lvl[xyz] = Numerics.max(gwbImage[xyz], cgbImage[xyz]+minthickness);
            }
            adjustLevelset(gwbImage, lvl);
            
            for (int t=0;t<repeats;t++) {
                
                System.out.println("repeat "+(t+1));
                
                // Run the adjustment for gwb
                lvl = fitBasicBoundarySigmoid(gwbImage, iterations, gwbContrastTypes);
                adjustLevelset(gwbImage, lvl);
            }
            
            // Push cgb boundary outward from gwb
            for (int xyz=0;xyz<nxyz;xyz++) {
                lvl[xyz] = Numerics.min(cgbImage[xyz], gwbImage[xyz]-minthickness);
            }
            adjustLevelset(cgbImage, lvl);
            
            for (int t=0;t<repeats;t++) {
                
                System.out.println("repeat "+(t+1));
            
                // Run the adjustment for cgb
                lvl = fitBasicBoundarySigmoid(cgbImage, iterations, cgbContrastTypes);
                adjustLevelset(cgbImage, lvl);
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
	
	public float[] fitBasicBoundarySigmoid(float[] levelset, int iter, byte[] contrastTypes) {

	    float delta = 0.001f;
	    float dist0 = 2.0f;
	    
	    int dist = Numerics.ceil(distance);
	    
	    // build smaller arrays for speed
	    int nspread = 0;
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz] && Numerics.abs(levelset[xyz])<spread) nspread++;
	    
	    // go over voxels close enough to the boundary
	    float[] reslevel = new float[nxyz];
	    float[] newlevel = new float[nspread];
	    int[] coord = new int[nspread];
	    float[][] interior = new float[nc][nspread];
	    float[][] exterior = new float[nc][nspread];
	    
	    int s=0;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        reslevel[xyz] = levelset[xyz];
	        if (mask[xyz] && Numerics.abs(levelset[xyz])<spread) {
                newlevel[s] = levelset[xyz];
                coord[s] = xyz;
                s++;
            }
	    }
	    // precompute region interior, exterior?
        for (s=0;s<nspread;s++) {
            for (int c=0;c<nc;c++) {
                float incount = 0.0f;
                float excount = 0.0f;
                int xyz = coord[s];
                for (int dx=-dist;dx<=dist;dx++) for (int dy=-dist;dy<=dist;dy++) for (int dz=-dist;dz<=dist;dz++) {
                    int dxyz = xyz + dx + nx*dy + nx*ny*dz;
                    if (mask[dxyz]) {
                        if ( (contrastTypes[c]==INCREASING && ( (reslevel[dxyz]>reslevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                           || (reslevel[dxyz]<=reslevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                || (contrastTypes[c]==DECREASING && ( (reslevel[dxyz]>reslevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                            || (reslevel[dxyz]<=reslevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                                || (contrastTypes[c]==BOTH) ) {
                                
                            float win = - Numerics.bounded(reslevel[dxyz]/dist0, -1.0f, 1.0f);
                            if (win<0) {
                                exterior[c][s] += win*win*contrastImages[c][dxyz];
                                excount += win*win;
                            }
                            if (win>0) {
                                interior[c][s] += win*win*contrastImages[c][dxyz];
                                incount += win*win;
                            }
                        }
                    }
                }
                // skip if one is empty
                if (incount>0 && excount>0) {
                    interior[c][s] /= incount;
                    exterior[c][s] /= excount;
                }
            }
        }

	    // multiple iterations of the offset within the same basis
	    for (int t=0;t<iter;t++) {
	        System.out.println("iteration "+(t+1));
            for (s=0;s<nspread;s++) {
                reslevel[coord[s]] = newlevel[s];
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
                }
                if (!valid) ninvalid++;
                else {    
                    for (int dx=-dist;dx<=dist;dx++) for (int dy=-dist;dy<=dist;dy++) for (int dz=-dist;dz<=dist;dz++) {
                        int dxyz = xyz + dx + nx*dy + nx*ny*dz;
                        if (mask[dxyz]) {
                            float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(reslevel[dxyz]/dist0)), delta, 1.0f-delta);
                            float wgtin = 0.0f;
                            float wgtex = 0.0f;
                            for (int c=0;c<nc;c++) {                                
                                if ( (contrastTypes[c]==INCREASING && ( (reslevel[dxyz]>reslevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) 
                                                                       || (reslevel[dxyz]<=reslevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) ) ) 
                                            || (contrastTypes[c]==DECREASING && ( (reslevel[dxyz]>reslevel[xyz] && contrastImages[c][dxyz]<=contrastImages[c][xyz]) 
                                                                        || (reslevel[dxyz]<=reslevel[xyz] && contrastImages[c][dxyz]>contrastImages[c][xyz]) ) ) 
                                            || (contrastTypes[c]==BOTH) ) {
                                                
                                    // oreder not important, symmetric values
                                    wgtin += Numerics.bounded((exterior[c][s]-contrastImages[c][dxyz])/(exterior[c][s]-interior[c][s]), delta, 1.0f-delta);
                                    wgtex += Numerics.bounded((contrastImages[c][dxyz]-interior[c][s])/(exterior[c][s]-interior[c][s]), delta, 1.0f-delta);
                                }
                            }
                            inbound += levelset[dxyz]*wgtx*wgtin;
                            insum += wgtx*wgtin;
                    
                            exbound += levelset[dxyz]*wgtx*wgtex;
                            exsum += wgtx*wgtex;
                                    
                            bdsum += wgtx;
                        }
                    }
                    // seems to be a good compromise, using the relative probabilities for in/out as spatial bias
                    if (bdsum>0 && insum>0 && exsum>0) {
                        float offset = 0.5f*(inbound/bdsum + exbound/bdsum)/(insum/bdsum + exsum/bdsum);
                        newlevel[s] = reslevel[xyz]-offset;
                        
                        maxdiff = Numerics.max(maxdiff,Numerics.abs(offset));
                    } else {
                        System.out.print("!");
                    }
                }
            }
            System.out.println(" invalid: "+ninvalid+" / "+nspread);
            System.out.println(" max difference: "+maxdiff);
        }
        return reslevel;
    }

    void adjustLevelset(float[] source, float[] target) {
        
        Gdm3d gdm = new Gdm3d(source, distance+2.0f, nx, ny, nz, rx, ry, rz,
						null, null, target, 0.0f, 0.0f, smoothness, 1.0f-smoothness,
						connectivity, lutdir);
		
		gdm.evolveNarrowBand(50, 0.01f);
		
		source = gdm.getLevelSet();
    }

}
