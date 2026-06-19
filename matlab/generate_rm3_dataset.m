%% generate_rm3_dataset.m  — RM3 fault dataset generator (PTO-first), WEC-Sim 7.0
% This is the script run on MATLAB Online to produce the real dataset.
% It writes a fresh wecSimInputFile.m per run (with Hs/Tp/PTO-damping baked in),
% runs wecSim, and exports one CSV per run with columns:
%   t, z1, z2, rel, vrel, Fpto, Ppto, Tmoor
%
% Verified WEC-Sim 7.0 API: pto(1).damping, output.bodies(i).time/position,
% output.ptos(i).velocity/forceInternalMechanics/powerInternalMechanics.
%
% Usage on MATLAB Online:
%   edit('/MATLAB Drive/generate_rm3_dataset.m'); paste; save; then
%   run('/MATLAB Drive/generate_rm3_dataset.m')
% Note: the cloned rm3.h5 is a git-lfs pointer — first run  cd examples/RM3/hydroData; bemio
clear; clc;

%% ---------- CONFIG ----------
wecSimRoot = '/MATLAB Drive/WEC-Sim';
rm3Dir     = fullfile(wecSimRoot,'examples','RM3');
outDir     = '/MATLAB Drive/wec_data/raw';

ptoNominal = 1200000;                              % nominal PTO damping [N/(m/s)]
endTime    = 400;                                  % full-length runs

faults   = {'healthy','pto_damping_loss'};
sevH = 1.0;  sevPTO = [0.8 0.6 0.4 0.2];           % all 4 severities
labelH = 0;  labelPTO = 1;
seaStates = [0.75 6.0; 0.75 9.0; 0.75 12.0;        % 3x3 grid = 9 sea states
             1.75 6.0; 1.75 9.0; 1.75 12.0;
             3.00 6.0; 3.00 9.0; 3.00 12.0];
% ----------------------------

addpath(genpath(wecSimRoot));
if ~exist(outDir,'dir'); mkdir(outDir); end

origInput = fullfile(rm3Dir,'wecSimInputFile.m');
backup    = fullfile(rm3Dir,'wecSimInputFile_STOCK_BACKUP.m');
if ~exist(backup,'file'); copyfile(origInput, backup); end
cleanupObj = onCleanup(@() restoreInput(backup, origInput));  % always restore

cd(rm3Dir);
runID = 0;  rows = {};

for fi = 1:numel(faults)
    fname = faults{fi};
    if strcmp(fname,'healthy'); sevs = sevH; label = labelH;
    else;                        sevs = sevPTO; label = labelPTO; end
    for sev = sevs
        if strcmp(fname,'pto_damping_loss'); ptoDamp = ptoNominal*sev;
        else;                                 ptoDamp = ptoNominal; end
        for ss = 1:size(seaStates,1)
            Hs = seaStates(ss,1); Tp = seaStates(ss,2);
            runID = runID + 1;
            fprintf('\n=== run %d | %s sev=%.2f Hs=%.2f Tp=%.2f ptoDamp=%.0f ===\n', ...
                    runID, fname, sev, Hs, Tp, ptoDamp);
            writeInputFile(origInput, Hs, Tp, ptoDamp, endTime);
            try
                wecSim;                          % reads the file we just wrote
                fn = sprintf('run_%05d__%s_sev%02d_Hs%03d_Tp%03d.csv', ...
                             runID, fname, round(sev*100), round(Hs*100), round(Tp*10));
                exportRun(output, fullfile(outDir,fn));
                rows(end+1,:) = {runID, fname, label, sev, Hs, Tp, fn};
                fprintf('  OK -> %s\n', fn);
            catch ME
                fprintf(2,'  FAILED: %s\n', ME.message);
            end
        end
    end
end

M = cell2table(rows, 'VariableNames', {'runID','fault','label','severity','Hs','Tp','file'});
writetable(M, fullfile(outDir,'manifest.csv'));
fprintf('\nDONE: %d runs. CSVs + manifest in %s\n', runID, outDir);

%% ---------- helpers ----------
function writeInputFile(path, Hs, Tp, ptoDamp, endTime)
    L = {
      '%% Auto-generated WEC-Sim input file (fault batch)'
      'simu = simulationClass();'
      'simu.simMechanicsFile = ''RM3.slx'';'
      'simu.mode = ''normal'';'
      'simu.explorer = ''off'';'
      'simu.startTime = 0;'
      'simu.rampTime = 100;'
      sprintf('simu.endTime = %g;', endTime)
      'simu.solver = ''ode4'';'
      'simu.dt = 0.1;'
      'waves = waveClass(''irregular'');'
      sprintf('waves.height = %g;', Hs)
      sprintf('waves.period = %g;', Tp)
      'waves.spectrumType = ''JS'';'
      'waves.phaseSeed = 1;'
      'body(1) = bodyClass(''hydroData/rm3.h5'');'
      'body(1).geometryFile = ''geometry/float.stl'';'
      'body(1).mass = ''equilibrium'';'
      'body(1).inertia = [20907301 21306090.66 37085481.11];'
      'body(2) = bodyClass(''hydroData/rm3.h5'');'
      'body(2).geometryFile = ''geometry/plate.stl'';'
      'body(2).mass = ''equilibrium'';'
      'body(2).inertia = [94419614.57 94407091.24 28542224.82];'
      'constraint(1) = constraintClass(''Constraint1'');'
      'constraint(1).location = [0 0 0];'
      'pto(1) = ptoClass(''PTO1'');'
      'pto(1).stiffness = 0;'
      sprintf('pto(1).damping = %g;', ptoDamp)
      'pto(1).location = [0 0 0];'
    };
    fid = fopen(path,'w'); fprintf(fid,'%s\n', L{:}); fclose(fid);
end

function exportRun(output, filepath)
    t    = output.bodies(1).time;
    z1   = output.bodies(1).position(:,3);
    z2   = output.bodies(2).position(:,3);
    rel  = z1 - z2;
    vrel = output.ptos(1).velocity(:,3);
    Fpto = output.ptos(1).forceInternalMechanics(:,3);
    Ppto = output.ptos(1).powerInternalMechanics(:,3);
    Tmoor = zeros(numel(t),1);                 % no mooring yet (Phase 1)
    T = table(t,z1,z2,rel,vrel,Fpto,Ppto,Tmoor);
    writetable(T, filepath);
end

function restoreInput(backup, origInput)
    if exist(backup,'file'); copyfile(backup, origInput); end
end
