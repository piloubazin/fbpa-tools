package nl.fullbrainpicture.algorithms;


import nl.fullbrainpicture.utilities.*;
import nl.fullbrainpicture.structures.*;
import nl.fullbrainpicture.libraries.*;

import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;

/**
 * Generates a mesh connector at the center of the boundary surface between two levelset surfaces
 * 
 */
public class LevelsetsToMeshConnector {
    
    // image containers
    private float[] levelset1;
    private float[] levelset2;
    private float length = 6.0f;
    private float side = 0.5f;
    private float distance = 1.0f;
    
    private boolean found=false;
    private float[] pointList;
    private int[] triangleList;
    
	private int nx=-1,ny=-1,nz=-1,nxyz=-1;
	private float rx, ry, rz;

	private static final int X=0;
	private static final int Y=1;
	private static final int Z=2;
	
	// create inputs
	public final void setLevelsetImage1(float[] val) { levelset1 = val; }
	public final void setLevelsetImage2(float[] val) { levelset2 = val; }

	public final void setConnectorLength(float val) { length = val; }
	public final void setConnectorSide(float val) { side = val; }
	public final void setBoundaryDistance(float val) { distance = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	public final boolean isBoundaryFound() { return found; }
    public final float[] getPointList() { return pointList; }
    public final int[] getTriangleList() { return triangleList; }
    
	public final void execute() {
	    
	    System.out.print("\nConnector Generation");
	    
	    // 1. Define boundary and find its centroid
	    float avg_x = 0.0f;
	    float avg_y = 0.0f;
	    float avg_z = 0.0f;
	    int avg_n = 0;
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
			int xyz = x+nx*y+nx*ny*z;
			if (levelset1[xyz]<distance && levelset2[xyz]<distance) {
			    avg_x += x;
			    avg_y += y;
			    avg_z += z;
			    avg_n++;
			}
		}
		if (avg_n==0) {
		    System.out.println(" no common boundary found: exiting");
		    found=false;
		    return;
		}
		found=true;
		
		avg_x /= avg_n;
		avg_y /= avg_n;
		avg_z /= avg_n;
		
	    // best gradient approximation among various choices (most regular)
		double I1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y, avg_z, nx, ny, nz);
		double Imx1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x-1.0f, avg_y, avg_z, nx, ny, nz);
		double Ipx1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x+1.0f, avg_y, avg_z, nx, ny, nz);
		double Imy1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y-1.0f, avg_z, nx, ny, nz);
		double Ipy1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y+1.0f, avg_z, nx, ny, nz);
		double Imz1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y, avg_z-1.0f, nx, ny, nz);
		double Ipz1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y, avg_z+1.0f, nx, ny, nz);
		
		double I2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y, avg_z, nx, ny, nz);
		double Imx2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x-1.0f, avg_y, avg_z, nx, ny, nz);
		double Ipx2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x+1.0f, avg_y, avg_z, nx, ny, nz);
		double Imy2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y-1.0f, avg_z, nx, ny, nz);
		double Ipy2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y+1.0f, avg_z, nx, ny, nz);
		double Imz2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y, avg_z-1.0f, nx, ny, nz);
		double Ipz2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y, avg_z+1.0f, nx, ny, nz);
		
		//if (Double.isNaN(I) || Double.isNaN(Imx) || Double.isNaN(Ipx) || Double.isNaN(Imy) || Double.isNaN(Ipy) || Double.isNaN(Imz) || Double.isNaN(Ipz))
		//	System.out.println("NaN: levelset ("+pt0[X]+", "+pt0[Y]+", "+pt0[Z]+")");
		
		double length1 = I1;

		double Dx1 = 0.5*(Ipx1-Imx1);
		double Dy1 = 0.5*(Ipy1-Imy1);
		double Dz1 = 0.5*(Ipz1-Imz1);
		
		double grad1 = FastMath.sqrt(Dx1*Dx1 + Dy1*Dy1 + Dz1*Dz1);
		
		double length2 = I2;

		double Dx2 = 0.5*(Ipx2-Imx2);
		double Dy2 = 0.5*(Ipy2-Imy2);
		double Dz2 = 0.5*(Ipz2-Imz2);
		
		double grad2 = FastMath.sqrt(Dx2*Dx2 + Dy2*Dy2 + Dz2*Dz2);
		
		double pt1x = avg_x;
		double pt1y = avg_y;
		double pt1z = avg_z;
		
		if (grad1 > 0.01) {
			// closed form approximation to the closest point on the N-th layer surface 
			// (accurate if close enough to approximate the surface by a plane)
			pt1x -= length1*Dx1/grad1;
			pt1y -= length1*Dy1/grad1;
			pt1z -= length1*Dz1/grad1;
		}

		double pt2x = avg_x;
		double pt2y = avg_y;
		double pt2z = avg_z;
		
		if (grad1 > 0.01) {
			// closed form approximation to the closest point on the N-th layer surface 
			// (accurate if close enough to approximate the surface by a plane)
			pt2x -= length2*Dx2/grad2;
			pt2y -= length2*Dy2/grad2;
			pt2z -= length2*Dz2/grad2;
		}

		avg_x = (float)(0.5*(pt1x+pt2x));
		avg_y = (float)(0.5*(pt1y+pt2y));
		avg_z = (float)(0.5*(pt1z+pt2z));
		
	    // 2. Estimate normal direction, other directions
		Imx1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x-1.0f, avg_y, avg_z, nx, ny, nz);
		Ipx1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x+1.0f, avg_y, avg_z, nx, ny, nz);
		Imy1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y-1.0f, avg_z, nx, ny, nz);
		Ipy1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y+1.0f, avg_z, nx, ny, nz);
		Imz1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y, avg_z-1.0f, nx, ny, nz);
		Ipz1 = ImageInterpolation.linearInterpolation(levelset1, 0.0f, avg_x, avg_y, avg_z+1.0f, nx, ny, nz);
		
		Imx2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x-1.0f, avg_y, avg_z, nx, ny, nz);
		Ipx2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x+1.0f, avg_y, avg_z, nx, ny, nz);
		Imy2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y-1.0f, avg_z, nx, ny, nz);
		Ipy2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y+1.0f, avg_z, nx, ny, nz);
		Imz2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y, avg_z-1.0f, nx, ny, nz);
		Ipz2 = ImageInterpolation.linearInterpolation(levelset2, 0.0f, avg_x, avg_y, avg_z+1.0f, nx, ny, nz);
		
		double Dx = 0.5*(Ipx1-Imx1) - 0.5*(Ipx2-Imx2);
		double Dy = 0.5*(Ipy1-Imy1) - 0.5*(Ipy2-Imy2);
		double Dz = 0.5*(Ipz1-Imz1) - 0.5*(Ipz2-Imz2);
		
		double grad = FastMath.sqrt(Dx*Dx + Dy*Dy + Dz*Dz);
		
		double T1x, T1y, T1z;
		double T2x, T2y, T2z;
	    if (Dx*Dx>Dy*Dy && Dx*Dx>Dz*Dz) {
	        T1x = 0.0 - Dy/grad*Dx/grad;
	        T1y = 1.0 - Dy/grad*Dy/grad;
	        T1z = 0.0 - Dy/grad*Dz/grad;
	        T2x = 0.0 - Dz/grad*Dx/grad;
	        T2y = 0.0 - Dz/grad*Dy/grad;
	        T2z = 1.0 - Dz/grad*Dz/grad;
	    } else if (Dy*Dy>Dz*Dz && Dy*Dy>Dx*Dx) {
		    T1x = 0.0 - Dz/grad*Dx/grad;
	        T1y = 0.0 - Dz/grad*Dy/grad;
	        T1z = 1.0 - Dz/grad*Dz/grad;
	        T2x = 1.0 - Dx/grad*Dx/grad;
	        T2y = 0.0 - Dx/grad*Dy/grad;
	        T2z = 0.0 - Dx/grad*Dz/grad;
	    } else {
	        T1x = 1.0 - Dx/grad*Dx/grad;
	        T1y = 0.0 - Dx/grad*Dy/grad;
	        T1z = 0.0 - Dx/grad*Dz/grad;
	        T2x = 0.0 - Dy/grad*Dx/grad;
	        T2y = 1.0 - Dy/grad*Dy/grad;
	        T2z = 0.0 - Dy/grad*Dz/grad;
	    }
	    double N1 = FastMath.sqrt(T1x*T1x + T1y*T1y + T1z*T1z);
	    double N2 = FastMath.sqrt(T2x*T2x + T2y*T2y + T2z*T2z);
	    
	    double T3x = 1.0/2.0*T1x/N1 + FastMath.sqrt(3.0)/2.0*T2x/N2;
	    double T3y = 1.0/2.0*T1y/N1 + FastMath.sqrt(3.0)/2.0*T2y/N2;
	    double T3z = 1.0/2.0*T1z/N1 + FastMath.sqrt(3.0)/2.0*T2z/N2;

	    double N3 = FastMath.sqrt(T3x*T3x + T3y*T3y + T3z*T3z);

	    // testing orthogonality
	    //System.out.println(" D*T1 = "+ (Dx*T1x+Dy*T1y+Dz*T1z)/(grad*N1));
	    //System.out.println(" D*T2 = "+ (Dx*T2x+Dy*T2y+Dz*T2z)/(grad*N2));
	    //System.out.println(" D*T3 = "+ (Dx*T3x+Dy*T3y+Dz*T3z)/(grad*N3));
	    
	    // 3. Generate connector mesh
	    pointList = new float[3*20];
	    triangleList = new int[3*36];
	    
	    // top section
	    pointList[3*0+X] = avg_x - length*(float)(Dx/grad);
	    pointList[3*0+Y] = avg_y - length*(float)(Dy/grad);
	    pointList[3*0+Z] = avg_z - length*(float)(Dz/grad);
	    
	    pointList[3*1+X] = avg_x - length*(float)(Dx/grad) - side*(float)(T1x/N1);
	    pointList[3*1+Y] = avg_y - length*(float)(Dy/grad) - side*(float)(T1y/N1);
	    pointList[3*1+Z] = avg_z - length*(float)(Dz/grad) - side*(float)(T1z/N1);
	    
	    pointList[3*2+X] = avg_x - length*(float)(Dx/grad) - side*(float)(T3x/N3);
	    pointList[3*2+Y] = avg_y - length*(float)(Dy/grad) - side*(float)(T3y/N3);
	    pointList[3*2+Z] = avg_z - length*(float)(Dz/grad) - side*(float)(T3z/N3);
	    
	    triangleList[3*0+0] = 0;
	    triangleList[3*0+1] = 1;
	    triangleList[3*0+2] = 2;
	    
	    pointList[3*3+X] = avg_x - length*(float)(Dx/grad) + side*(float)(T1x/N1) - side*(float)(T3x/N3);
	    pointList[3*3+Y] = avg_y - length*(float)(Dy/grad) + side*(float)(T1y/N1) - side*(float)(T3y/N3);
	    pointList[3*3+Z] = avg_z - length*(float)(Dz/grad) + side*(float)(T1z/N1) - side*(float)(T3z/N3);
	    
	    triangleList[3*1+0] = 0;
	    triangleList[3*1+1] = 2;
	    triangleList[3*1+2] = 3;
	    
	    pointList[3*4+X] = avg_x - length*(float)(Dx/grad) + side*(float)(T1x/N1);
	    pointList[3*4+Y] = avg_y - length*(float)(Dy/grad) + side*(float)(T1y/N1);
	    pointList[3*4+Z] = avg_z - length*(float)(Dz/grad) + side*(float)(T1z/N1);
	    
	    triangleList[3*2+0] = 0;
	    triangleList[3*2+1] = 3;
	    triangleList[3*2+2] = 4;
	    
	    pointList[3*5+X] = avg_x - length*(float)(Dx/grad) + side*(float)(T3x/N3);
	    pointList[3*5+Y] = avg_y - length*(float)(Dy/grad) + side*(float)(T3y/N3);
	    pointList[3*5+Z] = avg_z - length*(float)(Dz/grad) + side*(float)(T3z/N3);
	    
	    triangleList[3*3+0] = 0;
	    triangleList[3*3+1] = 4;
	    triangleList[3*3+2] = 5;
	    
	    pointList[3*6+X] = avg_x - length*(float)(Dx/grad) - side*(float)(T1x/N1) + side*(float)(T3x/N3);
	    pointList[3*6+Y] = avg_y - length*(float)(Dy/grad) - side*(float)(T1y/N1) + side*(float)(T3y/N3);
	    pointList[3*6+Z] = avg_z - length*(float)(Dz/grad) - side*(float)(T1z/N1) + side*(float)(T3z/N3);
	    
	    triangleList[3*4+0] = 0;
	    triangleList[3*4+1] = 5;
	    triangleList[3*4+2] = 6;
	    
	    triangleList[3*5+0] = 0;
	    triangleList[3*5+1] = 6;
	    triangleList[3*5+2] = 1;
	    
	    // bottom section
	    pointList[3*7+X] = avg_x + length*(float)(Dx/grad);
	    pointList[3*7+Y] = avg_y + length*(float)(Dy/grad);
	    pointList[3*7+Z] = avg_z + length*(float)(Dz/grad);
	    
	    pointList[3*8+X] = avg_x + length*(float)(Dx/grad) - side*(float)(T1x/N1);
	    pointList[3*8+Y] = avg_y + length*(float)(Dy/grad) - side*(float)(T1y/N1);
	    pointList[3*8+Z] = avg_z + length*(float)(Dz/grad) - side*(float)(T1z/N1);
	    
	    pointList[3*9+X] = avg_x + length*(float)(Dx/grad) - side*(float)(T3x/N3);
	    pointList[3*9+Y] = avg_y + length*(float)(Dy/grad) - side*(float)(T3y/N3);
	    pointList[3*9+Z] = avg_z + length*(float)(Dz/grad) - side*(float)(T3z/N3);
	    
	    triangleList[3*6+0] = 9;
	    triangleList[3*6+1] = 8;
	    triangleList[3*6+2] = 7;
	    
	    pointList[3*10+X] = avg_x + length*(float)(Dx/grad) + side*(float)(T1x/N1) - side*(float)(T3x/N3);
	    pointList[3*10+Y] = avg_y + length*(float)(Dy/grad) + side*(float)(T1y/N1) - side*(float)(T3y/N3);
	    pointList[3*10+Z] = avg_z + length*(float)(Dz/grad) + side*(float)(T1z/N1) - side*(float)(T3z/N3);
	    
	    triangleList[3*7+0] = 10;
	    triangleList[3*7+1] = 9;
	    triangleList[3*7+2] = 7;
	    
	    pointList[3*11+X] = avg_x + length*(float)(Dx/grad) + side*(float)(T1x/N1);
	    pointList[3*11+Y] = avg_y + length*(float)(Dy/grad) + side*(float)(T1y/N1);
	    pointList[3*11+Z] = avg_z + length*(float)(Dz/grad) + side*(float)(T1z/N1);
	    
	    triangleList[3*8+0] = 11;
	    triangleList[3*8+1] = 10;
	    triangleList[3*8+2] = 7;
	    
	    pointList[3*12+X] = avg_x + length*(float)(Dx/grad) + side*(float)(T3x/N3);
	    pointList[3*12+Y] = avg_y + length*(float)(Dy/grad) + side*(float)(T3y/N3);
	    pointList[3*12+Z] = avg_z + length*(float)(Dz/grad) + side*(float)(T3z/N3);
	    
	    triangleList[3*9+0] = 12;
	    triangleList[3*9+1] = 11;
	    triangleList[3*9+2] = 7;
	    
	    pointList[3*13+X] = avg_x + length*(float)(Dx/grad) - side*(float)(T1x/N1) + side*(float)(T3x/N3);
	    pointList[3*13+Y] = avg_y + length*(float)(Dy/grad) - side*(float)(T1y/N1) + side*(float)(T3y/N3);
	    pointList[3*13+Z] = avg_z + length*(float)(Dz/grad) - side*(float)(T1z/N1) + side*(float)(T3z/N3);
	    
	    triangleList[3*10+0] = 13;
	    triangleList[3*10+1] = 12;
	    triangleList[3*10+2] = 7;
	    
	    triangleList[3*11+0] = 8;
	    triangleList[3*11+1] = 13;
	    triangleList[3*11+2] = 7;
	    
	    // middle section and sides
	    pointList[3*14+X] = avg_x - side*(float)(T1x/N1);
	    pointList[3*14+Y] = avg_y - side*(float)(T1y/N1);
	    pointList[3*14+Z] = avg_z - side*(float)(T1z/N1);
	    
	    pointList[3*15+X] = avg_x - side*(float)(T3x/N3);
	    pointList[3*15+Y] = avg_y - side*(float)(T3y/N3);
	    pointList[3*15+Z] = avg_z - side*(float)(T3z/N3);
	    
	    pointList[3*16+X] = avg_x + side*(float)(T1x/N1) - side*(float)(T3x/N3);
	    pointList[3*16+Y] = avg_y + side*(float)(T1y/N1) - side*(float)(T3y/N3);
	    pointList[3*16+Z] = avg_z + side*(float)(T1z/N1) - side*(float)(T3z/N3);
	    
	    pointList[3*17+X] = avg_x + side*(float)(T1x/N1);
	    pointList[3*17+Y] = avg_y + side*(float)(T1y/N1);
	    pointList[3*17+Z] = avg_z + side*(float)(T1z/N1);
	    
	    pointList[3*18+X] = avg_x + side*(float)(T3x/N3);
	    pointList[3*18+Y] = avg_y + side*(float)(T3y/N3);
	    pointList[3*18+Z] = avg_z + side*(float)(T3z/N3);
	    
	    pointList[3*19+X] = avg_x - side*(float)(T1x/N1) + side*(float)(T3x/N3);
	    pointList[3*19+Y] = avg_y - side*(float)(T1y/N1) + side*(float)(T3y/N3);
	    pointList[3*19+Z] = avg_z - side*(float)(T1z/N1) + side*(float)(T3z/N3);
	    
	    triangleList[3*12+0] = 1;
	    triangleList[3*12+1] = 2;
	    triangleList[3*12+2] = 14;
	    
	    triangleList[3*13+0] = 2;
	    triangleList[3*13+1] = 14;
	    triangleList[3*13+2] = 15;
	    
	    triangleList[3*14+0] = 2;
	    triangleList[3*14+1] = 3;
	    triangleList[3*14+2] = 15;
	    
	    triangleList[3*15+0] = 3;
	    triangleList[3*15+1] = 15;
	    triangleList[3*15+2] = 16;
	    	    
	    triangleList[3*16+0] = 3;
	    triangleList[3*16+1] = 4;
	    triangleList[3*16+2] = 16;
	    
	    triangleList[3*17+0] = 4;
	    triangleList[3*17+1] = 16;
	    triangleList[3*17+2] = 17;
	    	    
	    triangleList[3*18+0] = 4;
	    triangleList[3*18+1] = 5;
	    triangleList[3*18+2] = 17;
	    
	    triangleList[3*19+0] = 5;
	    triangleList[3*19+1] = 17;
	    triangleList[3*19+2] = 18;
	    	    
	    triangleList[3*20+0] = 5;
	    triangleList[3*20+1] = 6;
	    triangleList[3*20+2] = 18;
	    
	    triangleList[3*21+0] = 6;
	    triangleList[3*21+1] = 18;
	    triangleList[3*21+2] = 19;
	    	    
	    triangleList[3*22+0] = 6;
	    triangleList[3*22+1] = 1;
	    triangleList[3*22+2] = 19;
	    
	    triangleList[3*23+0] = 1;
	    triangleList[3*23+1] = 19;
	    triangleList[3*23+2] = 14;
	    

	    triangleList[3*24+0] = 14;
	    triangleList[3*24+1] = 15;
	    triangleList[3*24+2] = 8;
	    
	    triangleList[3*25+0] = 15;
	    triangleList[3*25+1] = 8;
	    triangleList[3*25+2] = 9;
	    
	    triangleList[3*26+0] = 15;
	    triangleList[3*26+1] = 16;
	    triangleList[3*26+2] = 9;
	    
	    triangleList[3*27+0] = 16;
	    triangleList[3*27+1] = 9;
	    triangleList[3*27+2] = 10;
	    
	    triangleList[3*28+0] = 16;
	    triangleList[3*28+1] = 17;
	    triangleList[3*28+2] = 10;
	    
	    triangleList[3*29+0] = 17;
	    triangleList[3*29+1] = 10;
	    triangleList[3*29+2] = 11;
	    
	    triangleList[3*30+0] = 17;
	    triangleList[3*30+1] = 18;
	    triangleList[3*30+2] = 11;
	    
	    triangleList[3*31+0] = 18;
	    triangleList[3*31+1] = 11;
	    triangleList[3*31+2] = 12;
	    
	    triangleList[3*32+0] = 18;
	    triangleList[3*32+1] = 19;
	    triangleList[3*32+2] = 12;
	    
	    triangleList[3*33+0] = 19;
	    triangleList[3*33+1] = 12;
	    triangleList[3*33+2] = 13;
	    
	    triangleList[3*34+0] = 19;
	    triangleList[3*34+1] = 14;
	    triangleList[3*34+2] = 13;
	    
	    triangleList[3*35+0] = 14;
	    triangleList[3*35+1] = 13;
	    triangleList[3*35+2] = 8;
	    
	    return;
	}
	

}
