%% wecSimInputFile_template.m
% RM3 two-body point absorber input file, parameterised for batch fault-injection runs.
% Reads the following variables from the base workspace (set by run_batch.m):
%   Hs, Tp        : sea state (significant wave height, peak period)
%   faultName     : fault to inject (see config/faults.yaml)
%   sev           : severity multiplier for that fault
% Falls back to defaults if they are not set, so this file also runs standalone.
%
% Based on the stock WEC-Sim RM3 example. Keep this close to the tutorial so it
% runs unchanged on MATLAB Online once Gate 1 passes.

%% Pull batch parameters from the workspace (with safe defaults)
if ~exist('Hs','var'),        Hs = 2.0;            end
if ~exist('Tp','var'),        Tp = 8.0;            end
if ~exist('faultName','var'), faultName = 'healthy'; end
if ~exist('sev','var'),       sev = 1.0;           end

%% Simulation Data
simu = simulationClass();
simu.simMechanicsFile = 'RM3.slx';   % stock RM3 Simulink model
simu.solver = 'ode4';
simu.explorer = 'off';               % off for headless batch runs
simu.startTime = 0;
simu.rampTime = 100;
simu.endTime = 400;                  % keep modest for MATLAB Online batch speed
simu.dt = 0.1;
simu.cicEndTime = 30;

%% Wave Information — irregular (JONSWAP), driven by the sea-state matrix
waves = waveClass('irregular');
waves.height = Hs;                   % significant wave height (m)
waves.period = Tp;                   % peak period (s)
waves.spectrumType = 'JS';           % JONSWAP
waves.phaseSeed = 1;                 % fixed seed for reproducibility

%% Body Data
% Body 1: Float
body(1) = bodyClass('hydroData/rm3.h5');
body(1).geometryFile = 'geometry/float.stl';
body(1).mass = 'equilibrium';
body(1).inertia = [20907301 21306090.66 37085481.11];

% Body 2: Spar/Plate
body(2) = bodyClass('hydroData/rm3.h5');
body(2).geometryFile = 'geometry/plate.stl';
body(2).mass = 'equilibrium';
body(2).inertia = [94419614.57 94407091.24 28542224.82];

%% PTO and Constraint Parameters
% Floating (3DOF) constraint between the two bodies and ground
constraint(1) = constraintClass('Constraint1');
constraint(1).location = [0 0 0];

% Translational PTO (heave) — the channel a PTO fault degrades
pto(1) = ptoClass('PTO1');
pto(1).stiffness = 0;
pto(1).damping = 1200000;            % nominal PTO linear damping (N/(m/s))
pto(1).c = pto(1).damping;           % alias used by inject_fault (PTO damping coeff)
pto(1).location = [0 0 0];

%% Mooring — linear stiffness matrix (simplest; runs fast on MATLAB Online)
% Upgrade path: replace with MoorDyn for higher-fidelity line dynamics (guide §5.4).
mooring(1) = mooringClass('mooring');
mooring(1).matrix = diag([0 0 1e4 0 0 0]);  % heave-dominant linear mooring stiffness
mooring(1).location = [0 0 -200];

%% ---- Inject the fault (data-driven; healthy = no change) ----
pto(1).damping = pto(1).c;                       % sync before injection
[pto, mooring] = inject_fault(pto, mooring, faultName, sev);
pto(1).damping = pto(1).c;                       % apply injected damping to the active field

fprintf('[wecSimInputFile] fault=%s sev=%.2f Hs=%.2f Tp=%.2f | pto.c=%.0f moorK33=%.1f\n', ...
        faultName, sev, Hs, Tp, pto(1).c, mooring(1).matrix(3,3));
