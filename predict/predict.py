import re
from urllib.parse import urlsplit
import requests
import json
import asyncio
import aiohttp
from pathlib import Path


def get_istio_auth_session(url: str, username: str, password: str) -> dict:
    """
    Determine if the specified URL is secured by Dex and try to obtain a session cookie.
    WARNING: only Dex `staticPasswords` and `LDAP` authentication are currently supported
             (we default to using `staticPasswords` if both are enabled)

    :param url: Kubeflow server URL, including protocol
    :param username: Dex `staticPasswords` or `LDAP` username
    :param password: Dex `staticPasswords` or `LDAP` password
    :return: auth session information
    """
    # define the default return object
    auth_session = {
        "endpoint_url": url,  # KF endpoint URL
        "redirect_url": None,  # KF redirect URL, if applicable
        "dex_login_url": None,  # Dex login URL (for POST of credentials)
        "is_secured": None,  # True if KF endpoint is secured
        "session_cookie": None,  # Resulting session cookies in the form "key1=value1; key2=value2"
    }

    # use a persistent session (for cookies)
    with requests.Session() as s:
        ################
        # Determine if Endpoint is Secured
        ################
        resp = s.get(url, allow_redirects=True)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {url}"
            )

        auth_session["redirect_url"] = resp.url

        # if we were NOT redirected, then the endpoint is UNSECURED
        if len(resp.history) == 0:
            auth_session["is_secured"] = False
            return auth_session
        else:
            auth_session["is_secured"] = True

        ################
        # Get Dex Login URL
        ################
        redirect_url_obj = urlsplit(auth_session["redirect_url"])

        # if we are at `/auth?=xxxx` path, we need to select an auth type
        if re.search(r"/auth$", redirect_url_obj.path):
            #######
            # TIP: choose the default auth type by including ONE of the following
            #######

            # OPTION 1: set "staticPasswords" as default auth type
            redirect_url_obj = redirect_url_obj._replace(
                path=re.sub(r"/auth$", "/auth/local", redirect_url_obj.path)
            )
            # OPTION 2: set "ldap" as default auth type
            # redirect_url_obj = redirect_url_obj._replace(
            #     path=re.sub(r"/auth$", "/auth/ldap", redirect_url_obj.path)
            # )

        # if we are at `/auth/xxxx/login` path, then no further action is needed (we can use it for login POST)
        if re.search(r"/auth/.*/login$", redirect_url_obj.path):
            auth_session["dex_login_url"] = redirect_url_obj.geturl()
            # print(auth_session["dex_login_url"])

        # else, we need to be redirected to the actual login page
        else:
            # this GET should redirect us to the `/auth/xxxx/login` path
            resp = s.get(redirect_url_obj.geturl(), allow_redirects=True)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {redirect_url_obj.geturl()}"
                )

            # set the login url
            auth_session["dex_login_url"] = resp.url

        ################
        # Attempt Dex Login
        ################
        resp = s.post(
            auth_session["dex_login_url"],
            data={"login": username, "password": password},
            allow_redirects=True,
        )
        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials were probably invalid - "
                f"No redirect after POST to: {auth_session['dex_login_url']}"
            )

        # store the session cookies in a "key1=value1; key2=value2" string
        auth_session["session_cookie"] = "; ".join(
            [f"{c.name}={c.value}" for c in s.cookies]
        )
        auth_session["authservice_session"] = s.cookies.get("authservice_session")

    return auth_session


async def predict(url, headers, cookies, input_json, timeout):
    timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(cookies=cookies, timeout=timeout) as session:
        async with session.post(url=url, json=input_json, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                return {'status': response.status, 'message': await response.text()}


def get_json_files():
    input_data_path = Path("./json_data")
    input_datas = list(input_data_path.glob("*.json"))

    return input_datas


async def main():
    KUBEFLOW_ENDPOINT = "http://localhost:8080"
    KUBEFLOW_USERNAME = "user@example.com"
    KUBEFLOW_PASSWORD = "12341234"
    MODEL_NAME = "dino"
    SERVICE_HOSTNAME = "dino-s3.kubeflow-user-example-com.dinoai.com"
    PREDICT_ENDPOINT = f"{KUBEFLOW_ENDPOINT}/v1/models/{MODEL_NAME}:predict"

    ## 쿠키 생성
    _auth_session = get_istio_auth_session(
        url=KUBEFLOW_ENDPOINT, username=KUBEFLOW_USERNAME, password=KUBEFLOW_PASSWORD
    )
    cookies = {"authservice_session": _auth_session['authservice_session']}
    jar = requests.cookies.cookiejar_from_dict(cookies)


    ## 요청 헤더 생성
    headers = {"Host": SERVICE_HOSTNAME, "Content-Type": "application/json"}


    ## 테스트용 데이터 호출
    files = get_json_files()
    tasks = []
    for json_file in files:
        with open(json_file, 'r') as file:
            input_json = json.load(file)

            task = predict(PREDICT_ENDPOINT, headers, jar, input_json, timeout=600)
            tasks.append(task)

    responses = await asyncio.gather(*tasks)
    print(json.dumps(responses))

if __name__ == "__main__":
    asyncio.run(main())

