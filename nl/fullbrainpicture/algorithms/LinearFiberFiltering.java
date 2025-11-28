package nl.fullbrainpicture.algorithms;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;


public class LinearFiberFiltering {
	
	private float[] pvImage;
	private float[] diameterImage;
	private float[] thetaImage;
	private float[] lengthImage;
	
	private int[] parcellationImage;
	private float[] mgdmImage;
	
	private float[] probaImage;
	private int[] labelImage;
		
	private float[] thicknesses;
	private float[] angles;
	private float[] sizes;
	
	private int nc=0;
	private int nlabel=1;
	
	private float smooth=1.0f;
	private float scale=10.0f;
	
	private float diaSc;
	private float lenSc;
	private float angSc;
	
	
	// global variables
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);
	private static final	float	SQRT2PI = (float)FastMath.sqrt(2.0*(float)Math.PI);
	private static final	float	PI2 = (float)(Math.PI/2.0);
	private static final	float   L2N2=2.0f*(float)(FastMath.sqrt(2.0*(float)(FastMath.log(2.0))));
	private static final	float   INF=1e30f;
	private static final	float   ZERO=1e-30f;
	
	private static final int DIA=0;
	private static final int LEN=1;
	private static final int ANG=2;   
	
	private static final boolean debug=true;
	private static final boolean verbose=true;
	
	//set inputs
	public final void setPartialVolumeImage(float[] val) { pvImage = val;}
	public final void setLengthImage(float[] val) { lengthImage = val;}
	public final void setAngleImage(float[] val) { thetaImage = val; }
	public final void setDiameterImage(float[] val) { diameterImage = val;}
	
	public final void setParcellationImage(int[] val) { parcellationImage = val;}
		
	public final void setThicknesses(float[] val) { thicknesses = val; }
	public final void setAngles(float[] val) { angles = val; }
	public final void setSizes(float[] val) { sizes = val; }
	
	public final void setClusters(int val) { nc = val; }
	
	public final void setSmooth(float val) { smooth = val; }
	public final void setScale(float val) { scale = val; }
	
	// set generic inputs	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
			
	// create outputs
	public final float[] getProbaImage() { return probaImage;}
	public final int[] getLabelImage() { return labelImage;}

	public void execute(){
		BasicInfo.displayMessage("linear fiber filtering:\n");
		
		int nth = thicknesses.length;
		int nag = angles.length;
		int nsz = sizes.length;

		labelImage = new int[nxyz];
		
		if (parcellationImage!=null) {
			computeParcellationSurfaces(smooth, scale);
			probaImage = mgdmImage;
		} else {
			probaImage = new float[nxyz];
		}
		
		if (nc==0) {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) {
                int xyz = x+nx*y;
                if (pvImage[xyz]>0 && diameterImage[xyz]>0 && lengthImage[xyz]>0) {
                    labelImage[xyz] = 111;
                    // thickness
                    for (int n=0;n<nth;n++) {
                        if (diameterImage[xyz]>thicknesses[n]) {
                            labelImage[xyz] += 100;
                        }
                    }
                    // angle
                    double theta = thetaImage[xyz];
                    if (parcellationImage!=null) {
                        double gpx = 0.0;
                        if  (parcellationImage[xyz+1]==parcellationImage[xyz])
                            gpx += mgdmImage[xyz+1];
                        else
                            gpx -= mgdmImage[xyz+1];
                        
                        if  (parcellationImage[xyz-1]==parcellationImage[xyz])
                            gpx -= mgdmImage[xyz-1];
                        else
                            gpx += mgdmImage[xyz-1];
                        
                        double gpy = 0.0;
                        if  (parcellationImage[xyz+nx]==parcellationImage[xyz])
                            gpy += mgdmImage[xyz+nx];
                        else
                            gpy -= mgdmImage[xyz+nx];
                        
                        if  (parcellationImage[xyz-nx]==parcellationImage[xyz])
                            gpy -= mgdmImage[xyz-nx];
                        else
                            gpy += mgdmImage[xyz-nx];
                        
                        double np = FastMath.sqrt(gpx*gpx+gpy*gpy);
                        
                        double gax = FastMath.cos(theta/180.0*FastMath.PI);
                        double gay = FastMath.sin(theta/180.0*FastMath.PI);
                        
                        theta = 180.0/FastMath.PI*FastMath.acos(Numerics.abs(gax*gpx+gay*gpy)/np);
                    }
                    for (int n=0;n<nag;n++) {
                        if (theta>angles[n]) {
                            labelImage[xyz] += 10;
                        }
                    }
                    // size
                    for (int n=0;n<nsz;n++) {
                        if (lengthImage[xyz]>sizes[n]) {
                            labelImage[xyz] += 1;
                        }
                    }
                }
                    
            }
        } else {
            // multi-modal FCM approach 
            
            
            // convert from arbitrary angles to angles relating to structures
            computeParcellationAngles();
            
            // scales: given a priori for now, can be optimized further
            diaSc = 1.0f;
            lenSc = 5.0f;
            angSc = 30.0f;
            
            // find clusters per region, if given a parcellation
            if (parcellationImage!=null) {
                nlabel = ObjectLabeling.countLabels(parcellationImage, nx, ny, nz)-1;
            }
            
            float[][][] centroids = new float[nlabel][nc][3];
            // simple init: factor of the scales
            for (int l=0;l<nlabel;l++) for (int c=0;c<nc;c++) {
                centroids[l][c][DIA] = (c+1.0f)*diaSc;
                centroids[l][c][LEN] = (c+1.0f)*lenSc;
                centroids[l][c][ANG] = (c+1.0f)*angSc;
            }
            
            computeScales(centroids);
            
            float[][] mems = new float[nc][nxyz];
            computeMemberships(mems, centroids);
            
            float distance = 0.0f;
            int Niterations = 1;
            int maxIter = 50;
            float maxDist = 0.01f;
            boolean stop = false;
            if (Niterations >= maxIter) stop = true;
            while (!stop) {
                if (verbose) System.out.print("iteration " + Niterations + " (max: " + distance + ") \n");
			
                // update centroids
                computeCentroids(mems, centroids);
			
                // update scales (?)
                computeScales(centroids);
            
                // update membership
                distance = computeMemberships(mems, centroids);
			
                // check for segmentation convergence 
                Niterations++;
                if (Niterations > maxIter) stop = true;
                if (distance < maxDist) stop = true;            
            }
                
            // generate classification map
            labelImage = new int[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (pvImage[xyz]>0 && diameterImage[xyz]>0 && lengthImage[xyz]>0 && parcellationImage[xyz]>0) {
                    int best=0;
                    for (int c=1;c<nc;c++) {
                        if (mems[c][xyz]>mems[best][xyz]) best = c;
                    }
                    labelImage[xyz] = best+1;
                } else {
                    labelImage[xyz] = 0;
                }
            }            
        }
		
		return;
	}
	
	private void computeParcellationAngles() {
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) {
            int xyz = x+nx*y;
            if (pvImage[xyz]>0 && diameterImage[xyz]>0 && lengthImage[xyz]>0 && parcellationImage[xyz]>0) {
                
                double theta = thetaImage[xyz];
                if (parcellationImage!=null) {
                    double gpx = 0.0;
                    if  (parcellationImage[xyz+1]==parcellationImage[xyz])
                        gpx += mgdmImage[xyz+1];
                    else
                        gpx -= mgdmImage[xyz+1];
                    
                    if  (parcellationImage[xyz-1]==parcellationImage[xyz])
                        gpx -= mgdmImage[xyz-1];
                    else
                        gpx += mgdmImage[xyz-1];
                    
                    double gpy = 0.0;
                    if  (parcellationImage[xyz+nx]==parcellationImage[xyz])
                        gpy += mgdmImage[xyz+nx];
                    else
                        gpy -= mgdmImage[xyz+nx];
                    
                    if  (parcellationImage[xyz-nx]==parcellationImage[xyz])
                        gpy -= mgdmImage[xyz-nx];
                    else
                        gpy += mgdmImage[xyz-nx];
                    
                    double np = FastMath.sqrt(gpx*gpx+gpy*gpy);
                    
                    double gax = FastMath.cos(theta/180.0*FastMath.PI);
                    double gay = FastMath.sin(theta/180.0*FastMath.PI);
                    
                    theta = 180.0/FastMath.PI*FastMath.acos(Numerics.bounded(Numerics.abs(gax*gpx+gay*gpy)/np,-1.0,1.0));
                }
                thetaImage[xyz] = (float)theta;
            }
        }

	}
	
	private float computeMemberships(float[][] mems, float[][][] centroids) {
	    float distance,dist;
        float den,num;
        float neighbors, ngb;
        
        distance = 0.0f;

	    float[] prev = new float[nc];
	    
	    for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) {
	        int xyz = x+nx*y;
            if (pvImage[xyz]>0 && diameterImage[xyz]>0 && lengthImage[xyz]>0 && parcellationImage[xyz]>0) {
                den = 0.0f;
                int lb = parcellationImage[xyz]-1;
                for (int c=0;c<nc;c++) {
                    prev[c] = mems[c][xyz];
                }
                
                for (int c=0;c<nc;c++) {
                    // data term
                    num = 0.0f;
                    num += (diameterImage[xyz]-centroids[lb][c][DIA])*(diameterImage[xyz]-centroids[lb][c][DIA])/(diaSc*diaSc);
                    num += (lengthImage[xyz]-centroids[lb][c][LEN])*(lengthImage[xyz]-centroids[lb][c][LEN])/(lenSc*lenSc);
                    num += (thetaImage[xyz]-centroids[lb][c][ANG])*(thetaImage[xyz]-centroids[lb][c][ANG])/(angSc*angSc);
                    
                    // no spatial smoothing: doesn't make much sense here

                    // invert the result
                    if (num>ZERO) num = 1.0f/num;
                    else num = INF;

                    mems[c][xyz] = num;
                    den += num;
                }

                // normalization
                for (int c=0;c<nc;c++) {
                    mems[c][xyz] = mems[c][xyz]/den;

                    // compute the maximum distance
                    dist = Math.abs(mems[c][xyz]-prev[c]);
                    if (dist > distance) distance = dist;
                }
            }
        }
        return distance;
    }
    
    private void computeCentroids(float[][] mems, float[][][] centroids) {
        float[][] num = new float[nlabel][3];
        float[][] den = new float[nlabel][3];
        
        for (int c=0;c<nc;c++) {
            for (int l=0;l<nlabel;l++) for (int i=0;i<3;i++) {
                num[l][i] = 0.0f;
                den[l][i] = 0.0f;
            }
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (pvImage[xyz]>0 && diameterImage[xyz]>0 && lengthImage[xyz]>0 && parcellationImage[xyz]>0) {
                    int lb = parcellationImage[xyz]-1;
                    num[lb][DIA] += mems[c][xyz]*mems[c][xyz]*diameterImage[xyz];
                    den[lb][DIA] += mems[c][xyz]*mems[c][xyz];
                    num[lb][LEN] += mems[c][xyz]*mems[c][xyz]*lengthImage[xyz];
                    den[lb][LEN] += mems[c][xyz]*mems[c][xyz];
                    num[lb][ANG] += mems[c][xyz]*mems[c][xyz]*thetaImage[xyz];
                    den[lb][ANG] += mems[c][xyz]*mems[c][xyz];
                }
            }
            for (int l=0;l<nlabel;l++) for (int i=0;i<3;i++) {
               if (den[l][i]>0.0) {
                   centroids[l][c][i] = num[l][i]/den[l][i];
               } else {
                   centroids[l][c][i] = 0.0f;
               }
            }
        }
        if (verbose) {
            for (int l=0;l<nlabel;l++) {
                System.out.println("label: "+l);
                for (int i=0;i<3;i++) {
                    System.out.print(" centroids: ("+centroids[l][0][i]);
                    for (int c=1;c<nc;c++) System.out.print(", "+centroids[l][c][i]);
                    System.out.print(")\n");
                }
            }
		} 
		return;
    }
	
	
	private void computeScales(float[][][] centroids) {
	    float[] dist = new float[3];
        float den = 0.0f;
        
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) {
	        int xyz = x+nx*y;
            if (pvImage[xyz]>0 && diameterImage[xyz]>0 && lengthImage[xyz]>0 && parcellationImage[xyz]>0) {
                int lb = parcellationImage[xyz]-1;
                for (int c=0;c<nc;c++) {
                    
                    dist[DIA] += (diameterImage[xyz]-centroids[lb][c][DIA])*(diameterImage[xyz]-centroids[lb][c][DIA]);
                    dist[LEN] += (lengthImage[xyz]-centroids[lb][c][LEN])*(lengthImage[xyz]-centroids[lb][c][LEN]);
                    dist[ANG] += (thetaImage[xyz]-centroids[lb][c][ANG])*(thetaImage[xyz]-centroids[lb][c][ANG]);
                    
                    den++;
                }
            }
        }
        if (den>1.0f) {
            diaSc = (float)FastMath.sqrt(dist[DIA]/(den-1.0f));
            lenSc = (float)FastMath.sqrt(dist[LEN]/(den-1.0f));
            angSc = (float)FastMath.sqrt(dist[ANG]/(den-1.0f));
        }
        if (verbose) {
            System.out.print("scales: ("+diaSc+", "+lenSc+", "+angSc+")\n");
		} 
        return;
    }

	private void computeParcellationSurfaces(float smooth, float scale) {
				
		int nmgdm = 4;
		int nlb =  ObjectLabeling.countLabels(parcellationImage, nx, ny, nz);
                
		float[] proba = new float[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) proba[xyz] = 0.5f;
		
        // 3. Run MGDM!
        Mgdm2d mgdm = new Mgdm2d(parcellationImage, nx, ny, nlb, nmgdm, rx, ry, null, 
                                probaImage, parcellationImage,
                                0.0f, 0.5f/(1.0f+smooth), 0.5f*smooth/(1.0f+smooth), 0.0f, 
                                "no", null, true, scale);
        
        if (smooth>0.0f) {
        	mgdm.evolveNarrowBand(1500, 0.0001f);
        }
        
        // 4. copy the results
        parcellationImage = mgdm.reconstructedLabel(0);
		mgdmImage = new float[nx*ny*nz];
        for (int xyz=0;xyz<nx*ny*nz;xyz++) {
            mgdmImage[xyz] = mgdm.getFunctions()[0][xyz];
        }
        return;
    }

}
