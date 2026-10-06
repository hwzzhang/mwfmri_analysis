function make_fold_masks(manifest_path, output_root, image_root, contrast_pattern, fwe_alpha, min_extent, work_root, job_start, job_stop)
% Re-estimate one SPM12 one-sample t-test per training split.
%
% Model specification: positive group mean, implicit mask, no covariate, and
% no global scaling. Participants and output directories vary by CV fold.

arguments
    manifest_path (1,1) string
    output_root (1,1) string
    image_root (1,1) string
    contrast_pattern (1,1) string = "%s_con_0015.nii"
    fwe_alpha (1,1) double = 0.01  % FWE threshold; 0.05 can also be used
    min_extent (1,1) double = 16
    work_root (1,1) string = ""
    job_start (1,1) double = 1
    job_stop (1,1) double = -1
end

if ~(fwe_alpha > 0 && fwe_alpha < 1)
    error('fwe_alpha must be between 0 and 1.');
end
if min_extent < 0 || fix(min_extent) ~= min_extent
    error('min_extent must be a nonnegative integer.');
end

manifest = jsondecode(fileread(manifest_path));
jobs = manifest.jobs;
if job_stop < 0
    job_stop = numel(jobs);
end
if job_start < 1 || job_stop > numel(jobs) || job_start > job_stop || ...
        fix(job_start) ~= job_start || fix(job_stop) ~= job_stop
    error('Invalid inclusive job range [%g, %g] for %d jobs.', ...
        job_start, job_stop, numel(jobs));
end
if ~exist(output_root, 'dir')
    mkdir(output_root);
end
if strlength(work_root) == 0
    work_root = output_root;
end
if ~exist(work_root, 'dir')
    mkdir(work_root);
end
manifest_copy = fullfile(output_root, 'mask_manifest.json');
if ~exist(manifest_copy, 'file')
    copyfile(manifest_path, manifest_copy);
elseif ~strcmp(fileread(manifest_path), fileread(manifest_copy))
    error('Output root already contains a different mask_manifest.json.');
end

spm('defaults', 'FMRI');
spm_get_defaults('cmdline', true);
spm_jobman('initcfg');

for job_index = job_start:job_stop
    mask_id = string(jobs(job_index).mask_id);
    final_fold_dir = fullfile(output_root, mask_id);
    final_mask_path = fullfile(final_fold_dir, 'spmT_0001_fold_mask.nii.gz');
    uncompressed_mask_path = fullfile(final_fold_dir, 'spmT_0001_fold_mask.nii');
    provenance_path = fullfile(final_fold_dir, 'fold_mask_provenance.mat');
    if exist(final_mask_path, 'file') || exist(uncompressed_mask_path, 'file')
        if ~exist(provenance_path, 'file')
            error('Mask exists without provenance: %s', final_fold_dir);
        end
        saved = load(provenance_path);
        requested_ids = string(jobs(job_index).train_ids);
        if saved.fwe_alpha ~= fwe_alpha || saved.min_extent ~= min_extent || ...
                ~isequal(string(saved.train_ids(:)), requested_ids(:))
            error(['Existing mask provenance does not match the requested ' ...
                'participants/threshold: %s'], final_fold_dir);
        end
        fprintf('[%d/%d] %s already complete; skipping.\n', ...
            job_index, numel(jobs), mask_id);
        continue;
    end
    fold_dir = fullfile(work_root, mask_id);
    mask_path = fullfile(fold_dir, 'spmT_0001_fold_mask.nii');
    if exist(fullfile(fold_dir, 'SPM.mat'), 'file')
        error(['Partial SPM directory exists without its exported mask: %s. ' ...
            'Inspect it and use a clean output directory.'], fold_dir);
    end
    if ~exist(fold_dir, 'dir')
        mkdir(fold_dir);
    end
    if ~exist(final_fold_dir, 'dir')
        mkdir(final_fold_dir);
    end

    train_ids = string(jobs(job_index).train_ids);
    train_ids = train_ids(:);
    contrast_paths = fullfile(image_root, compose(contrast_pattern, train_ids));
    for path_index = 1:numel(contrast_paths)
        if ~exist(contrast_paths(path_index), 'file')
            error('Missing contrast image: %s', contrast_paths(path_index));
        end
    end
    contrast_images = cellstr(contrast_paths + ",1");

    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {char(fold_dir)};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = contrast_images;
    matlabbatch{1}.spm.stats.factorial_design.cov = ...
        struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.multi_cov = ...
        struct('files', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    spm_mat_path = char(fullfile(fold_dir, 'SPM.mat'));
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat_path};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    matlabbatch{3}.spm.stats.con.spmmat = {spm_mat_path};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Group Mean (Pos)';
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 1;
    spm_jobman('run', matlabbatch);

    resultbatch = {};
    resultbatch{1}.spm.stats.results.spmmat = {spm_mat_path};
    resultbatch{1}.spm.stats.results.conspec.titlestr = char(mask_id);
    resultbatch{1}.spm.stats.results.conspec.contrasts = 1;
    resultbatch{1}.spm.stats.results.conspec.threshdesc = 'FWE';
    resultbatch{1}.spm.stats.results.conspec.thresh = fwe_alpha;
    resultbatch{1}.spm.stats.results.conspec.extent = min_extent;
    resultbatch{1}.spm.stats.results.conspec.conjunction = 1;
    resultbatch{1}.spm.stats.results.conspec.mask.none = 1;
    resultbatch{1}.spm.stats.results.units = 1;
    resultbatch{1}.spm.stats.results.export{1}.binary.basename = 'fold_mask';
    spm_jobman('run', resultbatch);

    if ~exist(mask_path, 'file')
        error('SPM completed but the expected binary mask was not created: %s', mask_path);
    end
    provenance = struct();
    provenance.mask_id = mask_id;
    provenance.train_ids = train_ids;
    provenance.fwe_alpha = fwe_alpha;
    provenance.min_extent = min_extent;
    save(provenance_path, '-struct', 'provenance');
    compressed = gzip(mask_path, final_fold_dir);
    if isempty(compressed) || ~exist(final_mask_path, 'file')
        error('Failed to copy compressed final mask to: %s', final_mask_path);
    end
    if ~strcmp(char(fold_dir), char(final_fold_dir))
        % SPM leaves MATLAB's current directory inside the just-completed
        % model directory.  Move out before removing temporary intermediates,
        % otherwise the following job starts from a nonexistent cwd.
        cd(char(work_root));
        rmdir(fold_dir, 's');
    end
    fprintf('[%d/%d] %s complete (%d training participants).\n', ...
        job_index, numel(jobs), mask_id, numel(train_ids));
end
end
