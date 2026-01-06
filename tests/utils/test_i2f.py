import pytest
import reciprocalspaceship as rs
from pathlib import Path

from rsbooster.utils.i2f import (
    infer_intensity_column,
    infer_uncertainty_column,
    parse_arguments,
    main,
    OUTPUT_MTZ_DEFAULT_NAME,
    INFER_COLUMN_NAME,
)

@pytest.fixture
def fake_mtz_file(tmp_path: Path) -> Path:
    fake_file = tmp_path / "foo-for-test.mtz"
    fake_file.touch()
    return fake_file


class TestInferIntensityColumn:
    def test_finds_intensity_column(self, random_dataset: rs.DataSet) -> None:
        result = infer_intensity_column(random_dataset)
        assert result == "IMEAN"

    def test_raises_when_no_intensity_column(self, random_dataset: rs.DataSet) -> None:
        random_dataset = random_dataset.drop("IMEAN", axis="columns")
        with pytest.raises(ValueError, match="Could not infer intensity column"):
            infer_intensity_column(random_dataset)


class TestInferUncertaintyColumn:
    def test_finds_sigma_column_with_sig_prefix(
        self, random_dataset: rs.DataSet
    ) -> None:
        result = infer_uncertainty_column(random_dataset, "IMEAN")
        assert result == "SIGIMEAN"

    def test_raises_when_no_matching_sigma_column(
        self, random_dataset: rs.DataSet
    ) -> None:
        with pytest.raises(ValueError, match="Could not infer uncertainty column"):
            infer_uncertainty_column(random_dataset, "IOBS")

    def test_error_message(self, random_dataset: rs.DataSet) -> None:
        with pytest.raises(ValueError) as exc_info:
            infer_uncertainty_column(random_dataset, "IOBS")
        assert "SIGIOBS" in str(exc_info.value)  # candidate
        assert "IMEAN" in str(exc_info.value)  # available

    def test_raises_when_sigma_column_has_wrong_dtype(
        self, random_dataset: rs.DataSet
    ) -> None:
        random_dataset["SIGIMEAN"] = random_dataset["SIGIMEAN"].astype(rs.IntensityDtype())
        with pytest.raises(ValueError, match="Could not infer uncertainty column"):
            infer_uncertainty_column(random_dataset, "IMEAN")


class TestParseArguments:
    def test_minimal_arguments(self, fake_mtz_file: Path):
        args = parse_arguments([str(fake_mtz_file)])
        assert args.input_mtz == str(fake_mtz_file)
        assert args.output == OUTPUT_MTZ_DEFAULT_NAME
        assert args.inplace is False
        assert args.intensity_column == INFER_COLUMN_NAME
        assert args.uncertainties_column == INFER_COLUMN_NAME

    def test_nonexistent_input_file_raises(self, tmp_path):
        fake_path = tmp_path / "nonexistent.mtz"
        with pytest.raises(IOError, match="cannot find input file"):
            parse_arguments([str(fake_path)])

    def test_inplace_with_custom_output_raises(self, fake_mtz_file: Path):
        with pytest.raises(ValueError, match="inplace modification"):
            parse_arguments(
                [str(fake_mtz_file), "--inplace", "-o", "custom_output.mtz"]
            )

    def test_inplace_with_default_output_ok(self, fake_mtz_file: Path):
        args = parse_arguments([str(fake_mtz_file), "--inplace"])
        assert args.inplace is True
        assert args.output == OUTPUT_MTZ_DEFAULT_NAME


class TestMain:
    def test_main(self, tmp_path: Path, random_dataset: rs.DataSet) -> None:
        temp_mtz_file = tmp_path / "input.mtz"
        random_dataset.write_mtz(str(temp_mtz_file))

        output_file = tmp_path / "output.mtz"
        main(
            [
                str(temp_mtz_file),
                "-i", "IMEAN",
                "-u", "SIGIMEAN",
                "-o", str(output_file),
            ]
        )

        assert output_file.exists()
        result_ds = rs.read_mtz(str(output_file))
        assert len(result_ds) > 1

        assert "FW-F" in result_ds.columns
        assert "FW-SIGF" in result_ds.columns
        assert "FW-I" in result_ds.columns
        assert "FW-SIGI" in result_ds.columns
