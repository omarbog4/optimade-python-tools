""" This module contains a validator class that can be pointed
at an OPTIMADE implementation and validated against the pydantic
models in this package.

"""
# pylint: disable=import-outside-toplevel

import re
import sys
import logging
import random
import urllib.parse
from typing import Union, Tuple

try:
    import simplejson as json
except ImportError:
    import json

import requests
from fastapi.testclient import TestClient

from optimade.models import InfoResponse, EntryInfoResponse, IndexInfoResponse, DataType

from .utils import (
    ValidatorLinksResponse,
    ValidatorEntryResponseOne,
    ValidatorEntryResponseMany,
    ValidatorReferenceResponseOne,
    ValidatorReferenceResponseMany,
    ValidatorStructureResponseOne,
    ValidatorStructureResponseMany,
    Client,
    test_case,
    print_failure,
    print_notify,
    print_success,
    print_warning,
    ResponseError,
)


BASE_INFO_ENDPOINT = "info"
LINKS_ENDPOINT = "links"
VERSIONS_ENDPOINT = "versions"
NON_ENTRY_ENDPOINTS = (BASE_INFO_ENDPOINT, LINKS_ENDPOINT, VERSIONS_ENDPOINT)
REQUIRED_ENTRY_ENDPOINTS = ["references", "structures"]
TOP_LEVEL_PROPERTIES = ("id", "type", "attributes")

RESPONSE_CLASSES = {
    "references": ValidatorReferenceResponseMany,
    "references/": ValidatorReferenceResponseOne,
    "structures": ValidatorStructureResponseMany,
    "structures/": ValidatorStructureResponseOne,
    "info": InfoResponse,
    "links": ValidatorLinksResponse,
}
RESPONSE_CLASSES.update(
    {f"info/{entry}": EntryInfoResponse for entry in REQUIRED_ENTRY_ENDPOINTS}
)

REQUIRED_ENTRY_ENDPOINTS_INDEX = []
RESPONSE_CLASSES_INDEX = {
    BASE_INFO_ENDPOINT: IndexInfoResponse,
    LINKS_ENDPOINT: ValidatorLinksResponse,
}

PROPERTIES = {}
PROPERTIES["structures"] = {}
PROPERTIES["structures"]["MUST"] = ("id", "type", "structure_features")
PROPERTIES["structures"]["SHOULD"] = (
    "last_modified",
    "elements",
    "nelements" "elements_ratios",
    "chemical_formula_descriptive",
    "chemical_formula_reduced",
    "chemical_formula_anonymous",
    "dimension_types",
    "nperiodic_dimensions",
    "lattice_vectors",
    "cartesian_site_positions",
    "nsites",
    "species_at_sites",
    "species",
)
PROPERTIES["structures"]["OPTIONAL"] = (
    "immutable_id",
    "chemical_formula_hill",
    "assemblies",
)


class ImplementationValidator:
    """
    Class to call test functions on a particular OPTIMADE implementation.

    Uses the pydantic models in `optimade.models` to validate the
    response from the server and crawl through the available endpoints.

    Caution:
        Only works for current version of the specification as defined
        by `optimade.models`.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        client: Union[Client, TestClient] = None,
        base_url: str = None,
        verbosity: int = 0,
        page_limit: int = 5,
        max_retries: int = 5,
        run_optional_tests: bool = True,
        fail_fast: bool = False,
        as_type: str = None,
        index: bool = False,
    ):
        """ Set up the tests to run, based on constants in this module
        for required endpoints.

        """

        self.verbosity = verbosity
        self.max_retries = max_retries
        self.page_limit = page_limit
        self.index = index
        self.run_optional_tests = run_optional_tests
        self.fail_fast = fail_fast

        if as_type is None:
            self.as_type_cls = None
        elif self.index:
            if as_type not in RESPONSE_CLASSES_INDEX.keys():
                raise RuntimeError(
                    f"Provided as_type='{as_type}' not allowed for an Index meta-database."
                )
            self.as_type_cls = RESPONSE_CLASSES_INDEX[as_type]
        elif as_type in ("structure", "reference"):
            self.as_type_cls = RESPONSE_CLASSES[f"{as_type}s/"]
        else:
            self.as_type_cls = RESPONSE_CLASSES[as_type]

        if client is None and base_url is None:
            raise RuntimeError(
                "Need at least a URL or a client to initialize validator."
            )
        if base_url and client:
            raise RuntimeError("Please specify at most one of base_url or client.")
        if client:
            self.client = client
            self.base_url = self.client.base_url
        else:
            while base_url.endswith("/"):
                base_url = base_url[:-1]
            self.base_url = base_url
            self.client = Client(base_url, max_retries=self.max_retries)

        self._setup_log()
        self.expected_entry_endpoints = (
            REQUIRED_ENTRY_ENDPOINTS_INDEX if self.index else REQUIRED_ENTRY_ENDPOINTS
        )
        self.test_entry_endpoints = set(self.expected_entry_endpoints)

        self.response_classes = (
            RESPONSE_CLASSES_INDEX if self.index else RESPONSE_CLASSES
        )

        # some simple checks on base_url
        self.base_url_parsed = urllib.parse.urlparse(self.base_url)
        # only allow filters/endpoints if we are working in "as_type" mode
        if self.as_type_cls is None and self.base_url_parsed.query:
            raise SystemExit("Base URL not appropriate: should not contain a filter.")

        # if valid is True on exit, script returns 0 to shell
        # if valid is False on exit, script returns 1 to shell
        # if valid is None on exit, script returns 2 to shell, indicating an internal failure
        self.valid = None

        self._test_id_by_type = {}
        self._entry_info_by_type = {}

        self.success_count = 0
        self.failure_count = 0
        self.internal_failure_count = 0
        self.optional_success_count = 0
        self.optional_failure_count = 0
        self.failure_messages = []
        self.internal_failure_messages = []

    def _setup_log(self):
        """ Define stdout log based on given verbosity. """
        self._log = logging.getLogger("optimade").getChild("validator")
        self._log.handlers = []
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s | %(levelname)8s: %(message)s")
        )
        self._log.addHandler(stdout_handler)
        if self.verbosity == 0:
            self._log.setLevel(logging.CRITICAL)
        elif self.verbosity == 1:
            self._log.setLevel(logging.INFO)
        else:
            self._log.setLevel(logging.DEBUG)

    def validate_implementation(self):
        """ Run all the test cases of the implementation, or the single type test. """

        # If a single "as type" has been set, only run that test
        if self.as_type_cls is not None:
            self._log.info(
                "Validating response of %s with model %s",
                self.base_url,
                self.as_type_cls,
            )
            self.test_as_type()
            self.valid = not bool(self.failure_count)
            return

        # Test entire implementation
        print(f"Testing entire implementation at {self.base_url}...")
        print("\nMandatory tests:")
        self._log.debug("Testing base info endpoint of %s", BASE_INFO_ENDPOINT)

        # Save base info to construct new queries
        base_info = self.test_info_or_links_endpoints(BASE_INFO_ENDPOINT)
        self.base_info = base_info.dict()

        self.test_versions_endpoint(VERSIONS_ENDPOINT)

        self.available_json_endpoints = self.get_available_endpoints(base_info)

        # Test that entry info endpoints deserialize correctly
        for endp in self.test_entry_endpoints:
            entry_info_endpoint = f"{BASE_INFO_ENDPOINT}/{endp}"
            self._log.debug("Testing expected info endpoint %s", entry_info_endpoint)
            entry_info = self.test_info_or_links_endpoints(entry_info_endpoint)
            if entry_info:
                self._entry_info_by_type[endp] = entry_info.dict()

        # Use the _entry_info_by_type to construct filters on the relevant endpoints
        for endp in self.test_entry_endpoints:
            self.recurse_through_endpoint(endp)

        # Test that the results from multi-entry-endpoints obey e.g. page limits
        # and that all entries can be deserialized with the patched models
        #
        # These methods also set the test_ids for each type of entry, which are validated
        # in the next loop.
        for endp in self.test_entry_endpoints:
            self._log.debug("Testing multiple entry endpoint of %s", endp)
            self.test_multi_entry_endpoint(f"{endp}?page_limit={self.page_limit}")

        # Test that the single IDs scraped earlier work with the single entry endpoint
        for endp in self.test_entry_endpoints:
            self._log.debug("Testing single entry request of type %s", endp)
            self.test_single_entry_endpoint(endp)

        # Test that the links endpoint can be serialized correctly
        self._log.debug("Testing %s endpoint", LINKS_ENDPOINT)
        self.test_info_or_links_endpoints(LINKS_ENDPOINT)

        self.valid = not (bool(self.failure_count) or bool(self.internal_failure_count))

        self.print_summary()

    def recurse_through_endpoint(self, endp):
        entry_info = self._entry_info_by_type.get(endp)
        # self.check_entry_info(entry_info, endp)

        chosen_entry = self.get_archetypical_entry(endp)

        if not entry_info:
            raise ResponseError(
                f"Unable to generate filters for endpoint {endp}: entry info not found."
            )

        _impl_properties = entry_info["data"]["properties"]

        for prop in _impl_properties:
            # check support level of property
            prop_type = _impl_properties[prop]["type"]
            sortable = _impl_properties[prop]["sortable"]
            self.construct_queries_for_property(
                prop, prop_type, sortable, endp, chosen_entry
            )

    @test_case
    def get_archetypical_entry(self, endp):
        response = self.get_endpoint(endp)
        data_returned = response.json()["meta"]["data_returned"]
        assert data_returned > 0
        response = self.get_endpoint(
            f"{endp}?page_offset={random.randint(0, data_returned-1)}"
        )
        archetypical_entry = response.json()["data"][0]
        return archetypical_entry, f"set archetypical entry for {endp}"

    def construct_queries_for_property(
        self, prop, prop_type, sortable, endp, chosen_entry
    ):
        if prop_type in (DataType.INTEGER, DataType.FLOAT):
            self.construct_numerical_filters(
                prop, prop_type, sortable, endp, chosen_entry
            )
        elif prop_type == DataType.STRING:
            if prop != "type":
                self.construct_string_filters(
                    prop, prop_type, sortable, endp, chosen_entry
                )

    #     elif prop_type == DataType.LIST:
    #         self.construct_list_filters(prop, endp)
    #     elif prop_type == DataType.TIMESTAMP:
    #         self.construct_timestamp_filters(prop, endp)
    #     elif prop_type == DataType.STRING:
    # self.construct_string_filters(prop, endp)

    def construct_numerical_filters(
        self, prop, prop_type, sortable, endp, chosen_entry
    ):

        prop_value = chosen_entry["attributes"][prop]

        if prop_type == DataType.INTEGER:
            assert int(prop_value) == prop_value

        num_data_returned = {}
        inclusive_operators = ("=", "<=", ">=")
        exclusive_operators = ("!=", "<", ">")

        for operator in inclusive_operators + exclusive_operators:
            query = f"{prop} {operator} {prop_value}"
            response = self.get_endpoint(f"{endp}?filter={query}").json()
            num_data_returned[operator] = response["meta"]["data_returned"]

            excluded = operator in exclusive_operators
            # if we have all results on this page, check that the blessed ID is in the response
            if response["meta"]["data_returned"] <= len(response["data"]):
                assert excluded ^ (
                    chosen_entry["id"] in set(entry["id"] for entry in response["data"])
                )

            # check that at least the archetypical structure was returned
            if operator in inclusive_operators:
                assert num_data_returned[operator] >= 1

        assert num_data_returned["<="] >= num_data_returned["="]
        assert num_data_returned[">="] >= num_data_returned["="]

    def construct_string_filters(self, prop, prop_type, sortable, endp, chosen_entry):

        if prop == "id":
            prop_value = chosen_entry["id"]

        else:
            prop_value = chosen_entry["attributes"][prop]

        num_data_returned = {}
        inclusive_operators = ("=", "CONTAINS", "STARTS WITH", "ENDS WITH")
        exclusive_operators = ()

        for operator in inclusive_operators + exclusive_operators:
            query = f'{prop} {operator} "{prop_value}"'
            response = self.get_endpoint(f"{endp}?filter={query}").json()
            num_data_returned[operator] = response["meta"]["data_returned"]

            excluded = operator in exclusive_operators
            # if we have all results on this page, check that the blessed ID is in the response
            if response["meta"]["data_returned"] <= len(response["data"]):
                assert excluded ^ (
                    chosen_entry["id"] in set(entry["id"] for entry in response["data"])
                )

            # check that at least the archetypical structure was returned
            if operator in inclusive_operators:
                assert num_data_returned[operator] >= 1

        if prop == "id":
            assert num_data_returned["="] == 1

    def print_summary(self):
        if not self.valid:
            if self.failure_messages:
                print("\n\nFAILURES")
                print("========\n")
                for message in self.failure_messages:
                    print_failure(message[0])
                    for line in message[1]:
                        print_warning("\t" + line)

            if self.internal_failure_messages:
                print("\n\nINTERNAL FAILURES")
                print("=================\n")
                print(
                    "There were internal valiator failures associated with this run.\n"
                    "If this problem persists, please report it at:\n"
                    "https://github.com/Materials-Consortia/optimade-python-tools/issues/new.\n"
                )

                for message in self.internal_failure_messages:
                    print_notify(message[0])
                    for line in message[1]:
                        print_warning("\t" + line)

        if self.valid or (not self.valid and self.fail_fast):
            final_message = f"\n\nPassed {self.success_count} out of {self.success_count + self.failure_count + self.internal_failure_count} tests."
            if not self.valid:
                print_failure(final_message)
            else:
                print_success(final_message)

            if self.run_optional_tests:
                print(
                    f"Additionally passed {self.optional_success_count} out of "
                    f"{self.optional_success_count + self.optional_failure_count} optional tests."
                )

    def test_info_or_links_endpoints(self, request_str):
        """ Runs the test cases for the info endpoints. """
        response = self.get_endpoint(request_str)
        if response:
            deserialized = self.deserialize_response(
                response, self.response_classes[request_str]
            )
            if not deserialized:
                return response
            return deserialized
        return False

    def test_single_entry_endpoint(self, request_str):
        """ Runs the test cases for the single entry endpoints. """
        _type = request_str.split("?")[0]
        response_cls_name = _type + "/"
        if response_cls_name in self.response_classes:
            response_cls = self.response_classes[response_cls_name]
        else:
            self._log.warning(
                "Deserializing single entry response %s with generic response rather than defined endpoint.",
                _type,
            )
            response_cls = ValidatorEntryResponseOne
        if _type in self._test_id_by_type:
            test_id = self._test_id_by_type[_type]
            response = self.get_endpoint(f"{_type}/{test_id}")
            if response:
                self.deserialize_response(response, response_cls)

    def test_multi_entry_endpoint(self, request_str):
        """ Runs the test cases for the multi entry endpoints. """
        response = self.get_endpoint(request_str)
        _type = request_str.split("?")[0]
        if _type in self.response_classes:
            response_cls = self.response_classes[_type]
        else:
            self._log.warning(
                "Deserializing multi entry response from %s with generic response rather than defined endpoint.",
                _type,
            )
            response_cls = ValidatorEntryResponseMany
        deserialized = self.deserialize_response(response, response_cls)
        self.test_page_limit(response)
        self.get_single_id_from_multi_endpoint(deserialized)

    def test_versions_endpoint(self, request_str):
        """ Runs the test cases for the versions endpoint, which MUST exist for unversioned
        base URLs and MUST NOT exist for versioned base URLs.

        """
        expected_status_code = 200
        print(self.base_url_parsed.path)
        if (
            re.match(r"/v[0-9]+(\.[0-9]+){,2}/.*", self.base_url_parsed.path)
            is not None
        ):
            expected_status_code = 404

        response = self.get_endpoint(
            request_str, expected_status_code=expected_status_code
        )
        if response and expected_status_code == 200:
            self.test_versions_endpoint_content(response, request=request_str)

    @test_case
    def test_versions_endpoint_content(self, response):
        """ Check the content of the versions endpoint complies with the specification. """

        text_content = response.text.split("\n")
        headers = response.headers

        if text_content[0] != "version":
            raise ResponseError(
                f"First line of `/versions` response must be 'version', not {text_content[0]}"
            )

        if len(text_content) <= 1:
            raise ResponseError(
                f"No version numbers found in `/versions` response, only {text_content}"
            )

        for version in text_content[1:]:
            try:
                int(version)
            except ValueError:
                raise ResponseError(
                    f"Version numbers reported by `/versions` must be integers specifying the major version, not {version}"
                )

        content_type = headers.get("content-type")
        if content_type is not None:
            content_type = [_.replace(" ", "") for _ in content_type.split(";")]
        if not content_type or content_type[0].strip() != "text/csv":
            raise ResponseError(
                f"Incorrect content-type header {content_type} instead of 'text/csv'"
            )

        for type_parameter in content_type:
            if type_parameter == "header=present":
                break
        else:
            raise ResponseError(
                f"Missing 'header=present' parameter in content-type {content_type}"
            )

        return response, "`/versions` endpoint responded correctly."

    def test_as_type(self):
        response = self.get_endpoint("")
        if response:
            self._log.debug("Deserialzing response as type %s", self.as_type_cls)
            self.deserialize_response(response, self.as_type_cls)

    @test_case
    def test_page_limit(self, response, check_next_link: int = 5):
        """ Test that a multi-entry endpoint obeys the page limit by
        following pagination links up to a depth of `check_next_link`.

        Parameters:
            response (requests.Response): the response to test for page limit
                compliance.

        Keyword arguments:
            check_next_link (int): maximum recursion depth for following
                pagination links.

        Raises:
            ResponseError: if test fails in a predictable way.

        Returns:
            bool, str: True if the test was successful, with a string describing
                the success.

        """
        try:
            response = response.json()
        except (AttributeError, json.JSONDecodeError):
            raise ResponseError("Unable to test endpoint page limit.")

        try:
            num_entries = len(response["data"])
        except (KeyError, TypeError):
            raise ResponseError(
                "Response under `data` field was missing or had wrong type."
            )

        if num_entries > self.page_limit:
            raise ResponseError(
                f"Endpoint did not obey page limit: {num_entries} entries vs {self.page_limit} limit"
            )

        try:
            more_data_available = response["meta"]["more_data_available"]
        except KeyError:
            raise ResponseError("Field `meta->more_data_available` was missing.")

        if more_data_available and check_next_link:
            try:
                next_link = response["links"]["next"]
                if isinstance(next_link, dict):
                    next_link = next_link["href"]
            except KeyError:
                raise ResponseError(
                    "Endpoint suggested more data was available but provided no valid links->next link."
                )

            if not isinstance(next_link, str):
                raise ResponseError(
                    f"Unable to parse links->next {next_link!r} as a link."
                )

            self._log.debug("Following pagination link to %r.", next_link)
            next_response = self.get_endpoint(next_link)
            self.test_page_limit(next_response, check_next_link=check_next_link - 1)

        return (
            True,
            f"Endpoint obeyed page limit of {self.page_limit} by returning {num_entries} entries.",
        )

    @test_case
    def get_single_id_from_multi_endpoint(self, deserialized):
        """ Scrape an ID from the multi-entry endpoint to use as query
        for single entry endpoint.

        """
        if deserialized and deserialized.data:
            self._test_id_by_type[deserialized.data[0].type] = deserialized.data[0].id
            self._log.debug(
                "Set type %s test ID to %s",
                deserialized.data[0].type,
                deserialized.data[0].id,
            )
        else:
            raise ResponseError(
                "No entries found under endpoint to scrape ID from. "
                "This may be caused by previous errors, if e.g. the endpoint failed deserialization."
            )
        return (
            self._test_id_by_type[deserialized.data[0].type],
            f"successfully scraped test ID from {deserialized.data[0].type} endpoint",
        )

    @test_case
    def deserialize_response(self, response: requests.models.Response, response_cls):
        """ Try to create the appropriate pydantic model from the response. """
        if not response:
            raise ResponseError("Request failed")
        try:
            json_response = response.json()
        except json.JSONDecodeError:
            raise ResponseError(
                f"Unable to decode response as JSON. Response: {response}"
            )

        self._log.debug(
            f"Deserializing {json.dumps(json_response, indent=2)} as model {response_cls}"
        )
        return (
            response_cls(**json_response),
            "deserialized correctly as object of type {}".format(response_cls),
        )

    @test_case
    def get_available_endpoints(self, base_info):
        """ Try to get `entry_types_by_format` even if base info response could not be validated. """
        for _ in [0]:
            available_json_entry_endpoints = []
            try:
                available_json_entry_endpoints = base_info.data.attributes.entry_types_by_format.get(
                    "json"
                )
                break
            except Exception:
                self._log.warning(
                    "Info endpoint failed serialization, trying to manually extract entry_types_by_format."
                )

            if not base_info.json():
                raise ResponseError(
                    "Unable to get entry types from base info endpoint. "
                    f"This may most likely be attributed to a wrong request to the '{BASE_INFO_ENDPOINT}' endpoint."
                )

            try:
                available_json_entry_endpoints = base_info.json()["data"]["attributes"][
                    "entry_types_by_format"
                ]["json"]
                break
            except (KeyError, TypeError):
                raise ResponseError(
                    "Unable to get entry_types_by_format from unserializable base info response {}.".format(
                        base_info
                    )
                )
        else:
            raise ResponseError(
                "Unable to find any JSON entry types in entry_types_by_format"
            )

        if self.index and available_json_entry_endpoints != []:
            raise ResponseError(
                "No entry endpoint are allowed for an Index meta-database"
            )

        for non_entry_endpoint in NON_ENTRY_ENDPOINTS:
            if non_entry_endpoint in available_json_entry_endpoints:
                raise ResponseError(
                    f'Illegal entry "{non_entry_endpoint}" was found in entry_types_by_format"'
                )
        return (
            available_json_entry_endpoints,
            "successfully found available entry types in baseinfo",
        )

    @test_case
    def get_endpoint(
        self, request_str: str, expected_status_code: int = 200, optional: bool = False
    ) -> Tuple[requests.Response, str]:
        """ Gets the response from the endpoint specified by `request_str`.
        function is wrapped by the `test_case` decorator

        Parameters:
            request_str (str): the request to make to the client.

        Keyword arguments:
            expected_status_code (int): if the request responds with a different
                status code to this one, raise a ResponseError.
            optional (bool): whether the success of this test is optional.

        Returns:
            requests.Response: the response to the request (the `test_case` decorator
                swallows the success message returned in this function directly).

        """

        request_str = request_str.replace("\n", "")
        response = self.client.get(request_str)

        if response.status_code != expected_status_code:
            message = (
                f"Request to '{request_str}' returned HTTP code {response.status_code}."
            )
            message += "\nError(s):"
            for error in response.json().get("errors", []):
                message += f'\n  {error.get("title", "N/A")}: {error.get("detail", "N/A")} ({error.get("source", {}).get("pointer", "N/A")})'
            raise ResponseError(message)

        return response, f"received expected response: {response}."
