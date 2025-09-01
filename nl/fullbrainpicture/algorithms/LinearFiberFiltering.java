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
	
	private static final boolean smooth=false;
	
	// global variables
	private int nx, ny, nz, nc, nxyz;
	private float rx, ry, rz;
	
	// numerical quantities
	private static final	float	INVSQRT2 = (float)(1.0/FastMath.sqrt(2.0));
	private static final	float	INVSQRT3 = (float)(1.0/FastMath.sqrt(3.0));
	private static final	float	SQRT2 = (float)FastMath.sqrt(2.0);
	private static final	float	SQRT3 = (float)FastMath.sqrt(3.0);
	private static final	float	SQRT2PI = (float)FastMath.sqrt(2.0*(float)Math.PI);
	private static final	float	PI2 = (float)(Math.PI/2.0);
	private static final	float   L2N2=2.0f*(float)(FastMath.sqrt(2.0*(float)(FastMath.log(2.0))));
	
	// direction labeling		
	public	static	final	byte	X = 0;
	public	static	final	byte	Y = 1;
	public	static 	final 	byte 	XpY = 2;
	public	static 	final 	byte 	XmY = 3;
	public	static	final	byte	mX = 4;
	public	static	final	byte	mY = 5;
	public	static 	final 	byte 	mXpY = 6;
	public	static 	final 	byte 	mXmY = 7;
	
	public	static 	final 	byte 	NC = 4;
	public	static 	final 	byte 	NC2 = 8;
	
	private static final boolean debug=true;
	
	//set inputs
	public final void setPartialVolumeImage(float[] val) { pvImage = val;}
	public final void setLengthImage(float[] val) { lengthImage = val;}
	public final void setAngleImage(float[] val) { thetaImage = val; }
	public final void setDiameterImage(float[] val) { diameterImage = val;}
	
	public final void setParcellationImage(int[] val) { parcellationImage = val;}
		
	public final void setThicknesses(float[] val) { thicknesses = val; }
	public final void setAngles(float[] val) { angles = val; }
	public final void setSizes(float[] val) { sizes = val; }
	
	public final void setSmooth(boolean val) { smooth = val; }
	
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
			computeParcellationSurfaces(smooth);
			probaImage = mgdmImage;
		} else {
			probaImage = new float[nxyz];
		}
		
		for (int xyz=0;xyz<nxyz;xyz++) {
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
		
		return;
	}
	

	private void computeParcellationSurfaces(boolean smooth) {
				
		int nmgdm = 4;
		int nlb =  ObjectLabeling.countLabels(parcellationImage, nx, ny, nz);
                
		float[] proba = new float[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) proba[xyz] = 0.5f;
		
        // 3. Run MGDM!
        Mgdm2d mgdm = new Mgdm2d(parcellationImage, nx, ny, nlb, nmgdm, rx, ry, null, 
                                probaImage, parcellationImage,
                                0.0f, 0.1f, 0.4f, 0.0f, 
                                "no", null);
        
        if (smooth) {
        	mgdm.evolveNarrowBand(500, 0.001f);
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
