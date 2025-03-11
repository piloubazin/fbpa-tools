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
	private int iterations=1;
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
	public final void setIterations(int val) { iterations = val; }
	public final void setConnectivity(String val) { connectivity = val; }
	public final void setTopologyLUTdirectory(String val) { lutdir = val; }
	
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
	    
        float[] output = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (levelsetImage[xyz]<0) output[xyz] = 1.0f;
            else output[xyz] = 0.0f;
        }
        
        for (int t=0;t<iterations;t++) {
		    float[] newlevel = fitBasicBoundarySigmoid();
		    
		    adjustLevelset(newlevel);
		}
		
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (levelsetImage[xyz]<0) output[xyz] += 2.0f;
        }
        
        probaImage = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) {
            probaImage[xyz] = Numerics.bounded(levelsetImage[xyz],-distance, distance);
        }
        
        levelsetImage = output;
	    return;
	}
	
	public float[] fitBasicBoundarySigmoid() {

	    float delta = 0.001f;
	    float dist0 = 2.0f;
	    
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
	                    if ( (levelsetImage[dxyz]>levelsetImage[xyz] && contrastImage[dxyz]>contrastImage[xyz])
	                        || (levelsetImage[dxyz]<=levelsetImage[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) ) {
                            float win = - Numerics.bounded(levelsetImage[dxyz]/dist0, -1.0f, 1.0f);
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
                    /* not needed
                    // estimate stdev to deal with different variance within region
                    float indev = 0.0f;
                    float exdev = 0.0f;
                    for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                        int dxyz = dx+nx*dy+nx*ny*dz;
                        if (mask[dxyz]) {
                            if ( (levelsetImage[dxyz]>levelsetImage[xyz] && contrastImage[dxyz]>contrastImage[xyz])
                                || (levelsetImage[dxyz]<=levelsetImage[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) ) {
                                float win = - Numerics.bounded(levelsetImage[dxyz]/dist0, -1.0f, 1.0f);
                                if (win<0) {
                                    exdev += win*win*(contrastImage[dxyz]-exterior)*(contrastImage[dxyz]-exterior);
                                }
                                if (win>0) {
                                    indev += win*win*(contrastImage[dxyz]-interior)*(contrastImage[dxyz]-interior);
                                    incount += win*win;
                                }
                            } 
                        }
                    }
                    indev = (float)FastMath.sqrt(indev/incount);
                    exdev = (float)FastMath.sqrt(exdev/excount);
                    */
                    
                    // only take into account correct contrast values
                    if ( (contrastType==INCREASING && exterior>interior) 
                        || (contrastType==DECREASING && exterior<interior)
                        || (contrastType==BOTH) ) {
                    
                        /*
                        // offset
                        float offset = 0.0f;
                        float offcount = 0.0f;
                        for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                            int dxyz = dx+nx*dy+nx*ny*dz;
                            if (mask[dxyz]) {
                                if (levelsetImage[dxyz]>levelsetImage[xyz] && contrastImage[dxyz]>contrastImage[xyz]) {
                                    float wgty = Numerics.bounded((contrastImage[dxyz]-interior)/(exterior-interior), delta, 1.0f-delta)
                                                *Numerics.bounded((exterior-contrastImage[dxyz])/(exterior-interior), delta, 1.0f-delta);
                                    float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelsetImage[dxyz]/dist0)), delta, 1.0f-delta);
                    
                                    offset += levelsetImage[dxyz]*wgtx*wgty;
                                    offcount += wgtx*wgty;
                                } else if (levelsetImage[dxyz]<=levelsetImage[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) {
                                    float wgty = Numerics.bounded((contrastImage[dxyz]-interior)/(exterior-interior), delta, 1.0f-delta)
                                                *Numerics.bounded((exterior-contrastImage[dxyz])/(exterior-interior), delta, 1.0f-delta);
                                    float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelsetImage[dxyz]/dist0)), delta, 1.0f-delta);
                    
                                    offset += levelsetImage[dxyz]*wgtx*wgty;
                                    offcount += wgtx*wgty;
                                }
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
                            }*//*
                            newlevel[xyz] = levelsetImage[xyz]-offset;
                        }
                        */
                        // fit slope and offset together? seems unstable, bad starting point makes errors
                        /*
                        float avgphi=0.0f;
                        float avgimg=0.0f;
                        float avgnum=0.0f;
                        for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                            int dxyz = dx+nx*dy+nx*ny*dz;
                            if (mask[dxyz]) {
                                if ( (levelsetImage[dxyz]>levelsetImage[xyz] && contrastImage[dxyz]>contrastImage[xyz])
                                    || (levelsetImage[dxyz]<=levelsetImage[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) ) {
                                    float wgty = Numerics.bounded((contrastImage[dxyz]-interior)/(exterior-interior), delta, 1.0f-delta)
                                                *Numerics.bounded((exterior-contrastImage[dxyz])/(exterior-interior), delta, 1.0f-delta);
                                    float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelsetImage[dxyz]/dist0)), delta, 1.0f-delta);
                    
                                    avgphi += wgtx*wgty*levelsetImage[dxyz];
                                    avgimg += wgtx*wgty*contrastImage[dxyz];
                                    avgnum += wgtx*wgty;
                                } 
                            }
                        }
                        if (avgnum>0) {
                            avgphi /=avgnum;
                            avgimg /= avgnum;
                        }
                        float slope=0.0f;
                        float slden=0.0f;
                        for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                            int dxyz = dx+nx*dy+nx*ny*dz;
                            if (mask[dxyz]) {
                                if ( (levelsetImage[dxyz]>levelsetImage[xyz] && contrastImage[dxyz]>contrastImage[xyz])
                                    || (levelsetImage[dxyz]<=levelsetImage[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) ) {
                                    float wgty = Numerics.bounded((contrastImage[dxyz]-interior)/(exterior-interior), delta, 1.0f-delta)
                                                *Numerics.bounded((exterior-contrastImage[dxyz])/(exterior-interior), delta, 1.0f-delta);
                                    float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelsetImage[dxyz]/dist0)), delta, 1.0f-delta);
                    
                                    slope += wgtx*wgty*(levelsetImage[dxyz]-avgphi)*(contrastImage[dxyz]-avgimg);
                                    slden += wgtx*wgty*(levelsetImage[dxyz]-avgphi)*(levelsetImage[dxyz]-avgphi);
                                } 
                            }
                        }
                        if (slden>0) {
                            slope /= slden;
                        }
                        float offset = 0.0f;
                        if (slope!=0) {
                            offset = avgimg/slope - avgphi;
                        }
                        newlevel[xyz] = levelsetImage[xyz]+offset;
                        */
                        // offset as the location where probability in/out transition?
                        float inbound = 0.0f;
                        float exbound = 0.0f;
                        incount = 0.0f;
                        excount = 0.0f;
                        float bdcount = 0.0f;
                        for (int dx=x-dist;dx<=x+dist;dx++) for (int dy=y-dist;dy<=y+dist;dy++) for (int dz=z-dist;dz<=z+dist;dz++) {
                            int dxyz = dx+nx*dy+nx*ny*dz;
                            if (mask[dxyz]) {
                                if ( (levelsetImage[dxyz]>levelsetImage[xyz] && contrastImage[dxyz]>contrastImage[xyz]) 
                                    || (levelsetImage[dxyz]<=levelsetImage[xyz] && contrastImage[dxyz]<=contrastImage[xyz]) ) {
                                    float wgtin = Numerics.bounded((contrastImage[dxyz]-interior)/(exterior-interior), delta, 1.0f-delta);
                                    float wgtex = Numerics.bounded((exterior-contrastImage[dxyz])/(exterior-interior), delta, 1.0f-delta);
                                    
                                    float wgtx = Numerics.bounded((float)FastMath.exp(-0.5*Numerics.square(levelsetImage[dxyz]/dist0)), delta, 1.0f-delta);
                    
                                    inbound += levelsetImage[dxyz]*wgtx*wgtin;
                                    incount += wgtx*wgtin;
                    
                                    exbound += levelsetImage[dxyz]*wgtx*wgtex;
                                    excount += wgtx*wgtex;
                                    
                                    bdcount += wgtx;
                                }
                            }
                        }
                        // seems to be a good compromise, using the relative probabilities for in/out as spatial bias
                        if (bdcount>0) {
                            float offset = 0.5f*(inbound/bdcount + exbound/bdcount)/(incount/bdcount + excount/bdcount);
                            newlevel[xyz] = levelsetImage[xyz]-offset;
                        }
                        /* best so far
                        if (incount>0 && excount>0) {
                            float offset = 0.5f*(exdev*inbound/incount + indev*exbound/excount)/(exdev+indev);
                            newlevel[xyz] = levelsetImage[xyz]-offset;
                        }*/
                        /* grows systematically
                        if (incount+excount>0) {
                            float offset = 0.5f*(inbound + exbound)/(incount + exbound);
                            newlevel[xyz] = levelsetImage[xyz]-offset;
                        }*/
                    }
                }
            }
        }
        /*
        for (int xyz=0;xyz<nxyz;xyz++) if (newcount[xyz]>0) {
            newlevel[xyz] /= newcount[xyz];
        }*/
        
        return newlevel;
    }

    void adjustLevelset(float[] target) {
        
        Gdm3d gdm = new Gdm3d(levelsetImage, distance+2.0f, nx, ny, nz, rx, ry, rz,
						null, null, target, 0.0f, 0.0f, 0.5f, 0.5f,
						connectivity, lutdir);
		
		gdm.evolveNarrowBand(50, 0.01f);
		
		levelsetImage = gdm.getLevelSet();
    }
    
}
