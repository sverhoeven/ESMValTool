"""Fixes for FLUXNET obseravations."""
import iris

from ..fix import Fix


class reco(Fix):
    """Fixes for reco."""

    def fix_file(self, filepath, output_dir):
        """
        Apply fixes to the files prior to creating the cube.

        Should be used only to fix errors that prevent loading or can
        not be fixed in the cube (i.e. those related with missing_value
        and _FillValue or missing standard_name).

        Parameters
        ----------
        filepath: basestring
            file to fix.
        output_dir: basestring
            path to the folder to store the fix files, if required.

        Returns
        -------
        basestring
            Path to the corrected file. It can be different from the original
            filepath if a fix has been applied, but if not it should be the
            original filepath.

        """
        new_path = Fix.get_fixed_filepath(output_dir, filepath)
        cube = iris.load_cube(filepath)
        cube.standard_name = ('plant_respiration_carbon_flux')
        iris.save(cube, new_path)
        return new_path

    def fix_metadata(self, cubes):
        """
        Fix meta data.

        Fixes wrong coordinates.

        Parameters
        ----------
        cube: iris.cube.Cube

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        cube = cube.intersection(longitude=(-180.0, 180.0),
                                 latitude=(-90.0, 90.0))
        cube = cube.intersection(longitude=(0.0, 360.0))
        return [cube]
