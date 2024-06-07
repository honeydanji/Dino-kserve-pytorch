import re
from urllib.parse import urlsplit
import requests

def get_istio_auth_session(url: str, username: str, password: str) -> dict:
    auth_session = {
        "endpoint_url": url,
        "redirect_url": None,
        "dex_login_url": None,
        "is_secured": None,
        "session_cookie": None,
    }

    with requests.Session() as s:
        resp = s.get(url, allow_redirects=True)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {url}"
            )

        auth_session["redirect_url"] = resp.url

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
        print("authservice_session: ", s.cookies.get("authservice_session"))
        auth_session["authservice_session"] = s.cookies.get("authservice_session")

    return auth_session

if __name__ == "__main__":
    KUBEFLOW_ENDPOINT = "http://localhost:8080"
    KUBEFLOW_USERNAME = "user@example.com"
    KUBEFLOW_PASSWORD = "12341234"
    MODEL_NAME = "dino-s3"
    SERVICE_HOSTNAME = "dino-s3.kubeflow-user-example-com.svc.cluster.local"
    PREDICT_ENDPOINT = f"{KUBEFLOW_ENDPOINT}/v1/models/{MODEL_NAME}:predict"

    dino_input = {"instances": [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]} # 임시

    # 실제
    _auth_session = get_istio_auth_session(
        url=KUBEFLOW_ENDPOINT, username=KUBEFLOW_USERNAME, password=KUBEFLOW_PASSWORD
    )

    ## 쿠키를 얻어야 추론 요청이 가능하다.
    cookies = {"authservice_session": _auth_session['authservice_session']}

    jar = requests.cookies.cookiejar_from_dict(cookies)
    res = requests.post(
        url=PREDICT_ENDPOINT,
        headers={"Host": SERVICE_HOSTNAME, "Content-Type": "application/json"},
        cookies=jar,
        json=dino_input,
        timeout=200,
    )

    print("Status Code: ", res.status_code)
    # print("Response: ", res.text)
