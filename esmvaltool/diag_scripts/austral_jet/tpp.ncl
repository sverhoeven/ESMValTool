; #############################################################################
; austral_jet/tpp.ncl
; #############################################################################
; Author: Manuel Schlund (DLR, Germany)
; #############################################################################
;
; Description:
;     Simple script to calculate and save tropopause air pressure for the use
;     in other diagnostics (e.g. MLR models).
;
; Required diag_script_info attributes:
;     calc_type : Calculation type (mean/trend)
;     tag : Tag for the diagnostic.
;     lat_range : Latitude range.
;     season : Season which is considered.
;
; Optional diag_script_info attributes:
;     None
;
; Caveats:
;     None
;
; Modification history:
;     20190123-schl_ma: written.
;
; #############################################################################

load "interface_scripts/interface.ncl"

load "./diag_scripts/shared/statistics.ncl"


begin

  enter_msg(DIAG_SCRIPT, "")

  ; Check variables
  VARS = metadata_att_as_array(variable_info, "short_name")
  if (.not. any(VARS .eq. "ta")) then
    error_msg("f", DIAG_SCRIPT, "", "variable 'ta' not given")
  end if

  ; Input data
  INFO0 = select_metadata_by_name(input_file_info, "ta")
  DATASETS = metadata_att_as_array(INFO0, "dataset")
  DIM_DAT = dimsizes(DATASETS)

  ; Required attributes
  req_atts = (/"calc_type", "tag", "season", "lat_range"/)
  exit_if_missing_atts(diag_script_info, req_atts)
  req_var_atts = (/"var_type"/)
  do idat = 0, DIM_DAT - 1
    exit_if_missing_atts(INFO0[idat], req_var_atts)
  end do
  calc_type = diag_script_info@calc_type
  lat_range = diag_script_info@lat_range
  season = diag_script_info@season

  ; Create work directory
  system("mkdir -p " + config_user_info@work_dir)

end

begin

  ; Iterate over datasets
  do idat = 0, DIM_DAT - 1
    log_info(DATASETS(idat))
    info = INFO0[idat]
    A0 = read_data(info)

    ; Order plev: lowest pressure first (in hPa)
    pr = A0&plev
    pr = pr / 100.0
    if (pr(0) .eq. max(pr)) then
      pin = pr(::-1)
      tin = A0({time|:}, {lat|:}, {lon|:}, {plev|::-1})
    else
      pin = pr
      tin = A0({time|:}, {lat|:}, {lon|:}, {plev|:})
    end if

    ; Calculate tropopause pressure and copy metedata
    outvar = trop_wmo(pin, tin, 0, False)
    outvar!0 = A0!0
    outvar!1 = A0!2
    outvar!2 = A0!3
    outvar&time = A0&time
    outvar&lat = A0&lat
    outvar&lon = A0&lon
    outvar := area_operations(outvar, lat_range(0), lat_range(1), 0.0, \
                              360.0, "average", True)

    ; Extract season
    if (season .ne. "yearly") then
      outvar := time_operations(outvar, info@start_year, info@end_year, \
                                "extract", season, True)
    end if

    ; Calculation type
    if (calc_type .eq. "mean") then
      outvar := avg(outvar)
      outvar@var = "tpp"
      outvar@long_name = "Level of thermal tropopause according to WMO"
      outvar@units = "hPa"
    else if (calc_type .eq. "trend") then
      if (isatt(diag_script_info, "yearly_trend") .and. \
          diag_script_info@yearly_trend) then
        outvar := time_operations(outvar, info@start_year, info@end_year, \
                                  "average", "yearly", True)
        time = ispan(info@start_year, info@end_year, 1)
        units_suffix = " yr-1"
      else
        time = outvar&time
        units_suffix = " mon-1"
      end if
      rc = regline(time, outvar)
      outvar := rc
      outvar@var = "tpp_trend"
      outvar@long_name = "Level of thermal tropopause according to WMO " + \
        "(trend)"
      outvar@units = "hPa" + units_suffix
      delete([/time, rc, units_suffix/])
    else
      error_msg("f", DIAG_SCRIPT, "", "calc_type '" + calc_type + \
                "' not supported")
    end if
    end if

    ; Set other attributes
    outvar!0 = "time"
    outvar&time = avg((/info@start_year, info@end_year/))
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
    delete([/A0, pr, pin, tin, outvar, ncdf_outfile/])
  end do

  leave_msg(DIAG_SCRIPT, "")

end
