#pragma once
#include <iostream>
#include <vector>

class RADSoftGrid : public Scene
{
public:

	RADSoftGrid(const char* name) : Scene(name) {}

    float get_param_float(py::array_t<float> scene_params, int idx)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float out = ptr[idx];
        return out;
    }

    // params ordering: xpos, ypos, zpos, xsize, zsize, radius,
    // stretch, bend, shear, render_type, mass
	void Initialize(py::array_t<float> scene_params, int thread_idx=0)
    {
        auto ptr = (float *) scene_params.request().ptr;
	    float initX = ptr[0];
	    float initY = ptr[1];
	    float initZ = ptr[2];

		int dimx = (int)ptr[3];
        int dimy = (int)ptr[4];
		int dimz = (int)ptr[5];

        float radius = (float)ptr[6]; //0.025f

        // Elastic properties
        float stretchStiffness = ptr[7]; //0.9f;
        float bendStiffness = ptr[8]; //1.0f;
        float shearStiffness = ptr[9]; //0.9f;

        int render_type = ptr[10]; // 0: only points, 1: only mesh, 2: points + mesh
        float mass = float(ptr[11])/(dimx*dimy*dimz);   // avg bath towel is 500-700g

		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
        CreateSpringGrid(Vec3(initX, initY, initZ), dimx, dimy, dimz, radius,
                              phase, stretchStiffness, bendStiffness, shearStiffness,
                              0.0f, 1.0f/mass);
		g_numSubsteps = 4;
		g_params.numIterations = 30;

		g_params.dynamicFriction = 0.75f;
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);
		g_drawPoints = false;

        g_params.radius = radius*1.8f;
        // g_params.collisionDistance = 0.005f;

        // g_params.radius = radius; // particle-particle interaction radius - if centers of two particles closer than this, they interact
        // g_params.collisionDistance = radius; // particle-shape interaction
        g_params.collisionDistance = radius * 0.1; // particle-shape interaction
        // g_params.collisionDistance = radius * 0.05; // particle-shape interaction

        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >>1;
        g_drawSprings = false;

    }
};