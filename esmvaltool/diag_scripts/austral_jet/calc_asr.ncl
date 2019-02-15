; #############################################################################
; austral_jet/calc_asr.ncl
; #############################################################################
; Author: Manuel Schlund (DLR, Germany)
; #############################################################################
;
; Description:
;     Simple script to calculate and save absorbed shortwave correlation
;     gradient for the use in other diagnostics (e.g. MLR models).
;
; Required diag_script_info attributes:
;     tag : Tag for the diagnostic.
;     season : Season which is considered.
;
; Optional diag_script_info attributes:
;     None
;
; Caveats:
;     None
;
; Modification history:
;     20181204_A-schl_ma: written.
;
; #############################################################################

load "interface_scripts/interface.ncl"

load "./diag_scripts/shared/statistics.ncl"


begin

  enter_msg(DIAG_SCRIPT, "")

  ; Check variables
  VARS = metadata_att_as_array(variable_info, "short_name")
  if (.not. any(VARS .eq. "asr")) then
    error_msg("f", DIAG_SCRIPT, "", "variable 'asr' not given")
  end if

  ; Input data
  INFO0 = select_metadata_by_name(input_file_info, "asr")
  DATASETS = metadata_att_as_array(INFO0, "dataset")
  DIM_DAT = dimsizes(DATASETS)

  ; Required attributes
  req_atts = (/"tag", "season"/)
  exit_if_missing_atts(diag_script_info, req_atts)
  req_var_atts = (/"var_type"/)
  do idat = 0, DIM_DAT - 1
    exit_if_missing_atts(INFO0[idat], req_var_atts)
  end do
  season = diag_script_info@season

  ; Create work directory
  system("mkdir -p " + config_user_info@work_dir)

end

begin

  ; Iterate over datasets
  do idat = 0, DIM_DAT-1
    log_info(DATASETS(idat))
    info = INFO0[idat]
    A0 = read_data(info)

    ; Temporal average
    dummy0 = time_operations(A0, info@start_year, info@end_year, \
                             "average", season, True)

    ; Spatial average
    grad = new(dimsizes(dummy0&month), typeof(dummy0))
    do tt = 0, dimsizes(dummy0&month) - 1
      dummy2 = area_operations(dummy0(tt, :, :), -20.0, -50.0, 0.0, 360.0, \
                               "average", True)
      dummy3 = area_operations(dummy0(tt, :, :), -50.0, -90.0, 0.0, 360.0, \
                               "average", True)
      grad(tt) = dummy2 - dummy3
      delete([/dummy2, dummy3/])
    end do
    outvar = tofloat(avg(grad))
    delete([/dummy0, grad, A0/])

    ; Attributes
    outvar!0 = "time"
    outvar&time = avg((/info@start_year, info@end_year/))
    outvar@var = "asr_grad"
    outvar@long_name = "Absorbed shortwave radiation gradient"
    outvar@units = "W m-2"
    outvar@diag_script = DIAG_SCRIPT
    outvar@dataset = DATASETS(idat)
    if (isatt(info, "ensemble")) then
      outvar@ensemble = info@ensemble
    end if
    outvar@tag = diag_script_info@tag
    outvar@var_type = info@var_type
    outvar@project = info@project
    filename = systemfunc("basename " + info@filename)
    outvar@filename = config_user_info@work_dir + filename + "." + \
      outvar@var + ".nc"

    ; Save file
    ncdf_outfile = ncdf_write(outvar, outvar@filename)
    delete([/outvar, ncdf_outfile/])
  end do

  leave_msg(DIAG_SCRIPT, "")

end
