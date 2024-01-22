package nl.fullbrainpicture.algorithms;

import java.net.URL;

import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;


/*
 * @author Pierre-Louis Bazin
 */
public class ParcellationMgdmSmoothing {

	private int[] parcelImage = null;
	private float[] probaImage = null;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	private int 	iterationParam	=	500;
	private float 	changeParam		=	0.001f;
	private	float 	forceParam		= 	0.1f;
	private float 	curvParam		=	0.4f;
		
	private String 	topologyParam	=	"wcs";
	public static final String[] topoTypes = {"26/6", "6/26", "18/6", "6/18", "6/6", "wcs", "wco", "no"};
	private String	lutdir = null;
	
	// outputs
	private int[] segImage;
	private float[] mgdmImage;
	
	// create inputs
	public final void setParcellationImage(int[] val) { parcelImage = val; }
	public final void setProbabilityImage(float[] val) { probaImage = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
	
	public final void setDataWeight(float val) {forceParam = val; }
	public final void setCurvatureWeight(float val) { curvParam = val; }
	public final void setMaxIterations(int val) { iterationParam = val; }
	public final void setMinChange(float val) { changeParam = val; }

	public final void setTopology(String val) { topologyParam = val; }
	public final void setTopologyLUTdirectory(String val) { lutdir = val; }
	
	// create outputs
	public final int[] getSmoothedImage() { return segImage; }
	public final float[] getMgdmImage() { return mgdmImage; }

	public void execute(){

		
		int nmgdm = 4;
		int nlb =  ObjectLabeling.countLabels(parcelImage, nx, ny, nz);
                
        // 3. Run MGDM!
        Mgdm3d mgdm = new Mgdm3d(parcelImage, nx, ny, nz, nlb, nmgdm, rx, ry, rz, null, 
                                probaImage, parcelImage,
                                0.0f, forceParam, curvParam, 0.0f, 
                                topologyParam, lutdir);
        
        mgdm.evolveNarrowBand(iterationParam, changeParam);
        
        // 4. copy the results
        segImage = new int[nx*ny*nz];
		mgdmImage = new float[nx*ny*nz];
        for (int xyz=0;xyz<nx*ny*nz;xyz++) {
            segImage[xyz] = mgdm.getLabels()[0][xyz];
            mgdmImage[xyz] = mgdm.getFunctions()[0][xyz];
        }
        return;
    }
    
}
