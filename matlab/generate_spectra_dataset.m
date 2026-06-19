%% generate_spectra_dataset.m — unified 4-class dataset driven by REAL measured wave spectra
% Uses NDBC spectral-wave-density (swden) measured spectra via WEC-Sim spectrumImport, instead
% of idealized JONSWAP. 9 real spectra span calm->storm (Hs 1-7 m), selected in Python and
% bundled in real_spectra.mat (freqs[Hz], S_all[N,M] m^2/Hz, Hs, Tp).
%
% PREREQ on MATLAB Online:
%   - RM3MooringMatrix.slx already in examples/RM3 (Phase-1 step), rm3.h5 via bemio
%   - real_spectra.mat uploaded to /MATLAB Drive  (Home tab -> Upload)
% Usage: edit('/MATLAB Drive/generate_spectra_dataset.m'); paste; save; run(...)
clear; clc;

%% ---------- CONFIG ----------
wecSimRoot = '/MATLAB Drive/WEC-Sim';
rm3Dir     = fullfile(wecSimRoot,'examples','RM3');
outDir     = '/MATLAB Drive/wec_data/raw';
specFile   = '/MATLAB Drive/real_spectra.mat';     % uploaded file
ptoNom  = 1200000;   moorNom = 1e5;   endTime = 400;
specs = {
  'healthy',                0, 1.0;
  'pto_damping_loss',       1, [0.8 0.6 0.4 0.2];
  'mooring_stiffness_loss', 2, [0.75 0.5 0.25];
  'pto_plus_mooring',       4, 0.5;
};
% ----------------------------

assert(isfile(specFile), 'Upload real_spectra.mat to /MATLAB Drive first.');
SP = load(specFile);                 % freqs (M,1), S_all (N,M), Hs (N,1), Tp (N,1)
nSpec = size(SP.S_all, 1);

addpath(genpath(wecSimRoot));
if ~exist(outDir,'dir'); mkdir(outDir); end
delete(fullfile(outDir,'run_*.csv'));
if isfile(fullfile(outDir,'manifest.csv')); delete(fullfile(outDir,'manifest.csv')); end
origInput = fullfile(rm3Dir,'wecSimInputFile.m');
backup    = fullfile(rm3Dir,'wecSimInputFile_STOCK_BACKUP.m');
if ~exist(backup,'file'); copyfile(origInput, backup); end
cleanupObj = onCleanup(@() restoreInput(backup, origInput));

cd(rm3Dir);  runID = 0;  rows = {};
for si = 1:size(specs,1)
    fname = specs{si,1};  label = specs{si,2};  sevs = specs{si,3};
    for sev = sevs
        ptoDamp = ptoNom;  moorK = moorNom;
        if any(strcmp(fname,{'pto_damping_loss','pto_plus_mooring'}));       ptoDamp = ptoNom*sev;  end
        if any(strcmp(fname,{'mooring_stiffness_loss','pto_plus_mooring'})); moorK   = moorNom*sev; end
        for k = 1:nSpec
            Hs = SP.Hs(k);  Tp = SP.Tp(k);  runID = runID + 1;
            % write this spectrum to a 2-col [freq_Hz, S] file for spectrumImport
            writematrix([SP.freqs(:), SP.S_all(k,:)'], fullfile(rm3Dir,'spectrum_tmp.txt'), 'Delimiter','tab');
            fprintf('\n=== run %d | %s sev=%.2f spec%d Hs=%.2f Tp=%.2f ptoDamp=%.0f moorK=%.0f ===\n', ...
                    runID, fname, sev, k, Hs, Tp, ptoDamp, moorK);
            writeInputFile(origInput, ptoDamp, moorK, endTime);
            try
                wecSim;
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
fprintf('\nDONE: %d runs (real measured spectra). CSVs in %s\n', runID, outDir);

%% ---------- helpers ----------
function writeInputFile(path, ptoDamp, moorK, endTime)
    L = {
      'simu = simulationClass();'
      'simu.simMechanicsFile = ''RM3MooringMatrix.slx'';'
      'simu.mode = ''normal'';  simu.explorer = ''off'';'
      'simu.startTime = 0;  simu.rampTime = 100;'
      sprintf('simu.endTime = %g;', endTime)
      'simu.solver = ''ode4'';  simu.dt = 0.1;'
      'waves = waveClass(''spectrumImport'');'
      'waves.spectrumFile = ''spectrum_tmp.txt'';'     % written each run, read from cwd
      'waves.phaseSeed = 1;'
      'body(1) = bodyClass(''hydroData/rm3.h5'');  body(1).geometryFile = ''geometry/float.stl'';'
      'body(1).mass = ''equilibrium'';  body(1).inertia = [20907301 21306090.66 37085481.11];'
      'body(2) = bodyClass(''hydroData/rm3.h5'');  body(2).geometryFile = ''geometry/plate.stl'';'
      'body(2).mass = ''equilibrium'';  body(2).inertia = [94419614.57 94407091.24 28542224.82];'
      'body(2).initial.displacement = [0 0 -0.21];'
      'constraint(1) = constraintClass(''Constraint1'');  constraint(1).location = [0 0 0];'
      'pto(1) = ptoClass(''PTO1'');  pto(1).stiffness = 0;'
      sprintf('pto(1).damping = %g;', ptoDamp)
      'pto(1).location = [0 0 0];'
      'mooring(1) = mooringClass(''mooring'');'
      'mooring(1).matrix.stiffness = zeros(6,6);'
      sprintf('mooring(1).matrix.stiffness(1,1) = %g;', moorK)
      'mooring(1).matrix.damping = zeros(6,6);  mooring(1).matrix.preTension = zeros(1,6);'
    };
    fid = fopen(path,'w'); fprintf(fid,'%s\n', L{:}); fclose(fid);
end
function exportRun(output, filepath)
    t    = output.bodies(1).time;
    z1   = output.bodies(1).position(:,3);  z2 = output.bodies(2).position(:,3);
    rel  = z1 - z2;  vrel = output.ptos(1).velocity(:,3);
    Fpto = output.ptos(1).forceInternalMechanics(:,3);
    Ppto = output.ptos(1).powerInternalMechanics(:,3);
    x1   = output.bodies(1).position(:,1);  x2 = output.bodies(2).position(:,1);
    try
        Tmoor = output.mooring(1).forceMooring(:,1);
    catch
        Tmoor = zeros(numel(t),1);
    end
    T = table(t,z1,z2,rel,vrel,Fpto,Ppto,x1,x2,Tmoor);
    writetable(T, filepath);
end
function restoreInput(backup, origInput)
    if exist(backup,'file'); copyfile(backup, origInput); end
end
